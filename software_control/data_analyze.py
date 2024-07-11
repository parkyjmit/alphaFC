import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from PyEIS import EIS_exp, Parameters
import sigfig
from time import sleep


class Analyzer:
    def __init__(self, data_dir, technique, sample_area=1, **kwargs):
        self.sample_area = sample_area

        if technique == 'LSV':
            self.data = self.load_data(data_dir)
            assert 'I/mA' in self.data.columns
            assert 'Ewe/V' in self.data.columns
            self.technique = 'LSV'
        elif technique == 'CA':
            self.data = self.load_data(data_dir)
            assert 'time/s' in self.data.columns
            assert 'I/mA' in self.data.columns
            assert 'Ewe/V' in self.data.columns
            self.technique = 'CA'
        elif technique == 'EIS':
            path, data = data_dir.rsplit('/', 1)
            self.eis_data = EIS_Analyzer(path, [data])
            self.technique = 'EIS'

    @staticmethod
    def load_data(data_dir):
        while True:
            try:
                return pd.read_csv(
                    data_dir,
                    sep=';',
                )
            except FileNotFoundError as e:
                print(e, 'Retrying in 30s...')
                sleep(30)
                continue

    @staticmethod
    # get the closest potential value given a specific current value, consider multi modal LSV curve
    def get_closest_value(look_up_value, df, look_up_col_name, return_col_name):
        # reverse the df to efficiently find the last up going curve
        df_rev = df.iloc[::-1]
        for i, value in df_rev[look_up_col_name].items():
            if value - look_up_value < 0:
                return df[return_col_name].iloc[i:i + 2].mean()

    # get the maximum current from LSV data
    def get_max_i(self):
        assert self.technique == 'LSV'
        return round(self.data['I/mA'].max(), 1)

    # get the alkline OER overpotential in the data
    def get_alkline_oer_overpotential(self):
        assert self.technique == 'LSV'

        # overpotential defined as potential when current is 10mA/cm^2
        I_overpotential = 10 * self.sample_area
        # target value is 10mA/cm^2, ref potential 0.924
        observed_potential = self.get_closest_value(I_overpotential, self.data, 'I/mA', 'Ewe/V')
        overpotential = observed_potential + 0.924 - 1.23
        return round(overpotential, 4)

    # get the tafel slope from LSV data
    def get_tafel_slope(self, I_low=10, I_high=100):
        assert self.technique == 'LSV'

        p_low = self.get_closest_value(I_low * self.sample_area, self.data, 'I/mA', 'Ewe/V')
        p_high = self.get_closest_value(I_high * self.sample_area, self.data, 'I/mA', 'Ewe/V')
        onset_data = self.data[(self.data['Ewe/V'] > p_low) & (self.data['Ewe/V'] < p_high)]
        if len(onset_data) == 0:
            return np.nan
        else:
            onset_data = onset_data.assign(log_I=lambda x: np.log10(x['I/mA']))
            x = onset_data.loc[:, 'log_I'].values.reshape(-1, 1)
            y = onset_data.loc[:, 'Ewe/V'].values
            model = LinearRegression()
            model.fit(x, y)
            return round(model.coef_[0], 4), round(model.score(x, y), 3)

    # get the total energy output from CA data
    def get_total_energy(self, ref_potential, counter_potential):
        assert self.technique == 'CA'

        # working potential vs RHE
        working_potential = self.data['Ewe/V'] - ref_potential  # -0.263

        # voltage output calculation, create a pd series, calculate abs between working potential and counter potential
        voltage_output = working_potential.apply(lambda x: abs(x - counter_potential))  # ORR

        # calculate the power at each time frame
        power = voltage_output * self.data['I/mA']

        # calculate the energy output by integrating the power over time
        time_diff = self.data['time/s'].diff()
        energy_output = (power * time_diff).cumsum()

        return round(energy_output.iloc[-1], 1)

    def get_EIS_fitting_result(self, circuit, param_df):
        assert self.technique == 'EIS'
        return self.eis_data.get_fitting_result(circuit, param_df)


class EIS_Analyzer(EIS_exp):
    def __init__(self, path, data):
        """
        :param path: data storage path
        :param data: data file name
        """
        super().__init__(f'{path}/', data)

    def get_fitting_result(self, circuit, param_df):
        """
        :param circuit: ecm model, e.g. 'R-RQ'
        :param param_df: e.g.
            pd.DataFrame(columns=['name', 'value', 'min', 'max'], data=[
                    ['Rs', 50, 0.1, 100],
                    ['R', 200, 1, 1000],
                    ['Q', 1, 1e-4, 10],
                    ['n', 0.8, 0.5, 2]
                ]
            )
        """
        fit_params = Parameters()
        # add each parameter value, min, max to fit_params from param_df
        for i, row in param_df.iterrows():
            fit_params.add(row['name'], value=row['value'], min=row['min'], max=row['max'])
        self.EIS_fit(fit_params, circuit)
        fit_result = {
            param_name: [sigfig.round(self.__getattribute__(f'fit_{param_name}')[0], 4)]
            for param_name in param_df['name']
        }
        return pd.DataFrame.from_dict(fit_result)
