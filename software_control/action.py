from software_control.software import Software
from software_control.data_analyze import Analyzer, EIS_Analyzer
import pandas as pd
from ml import utils
from time import sleep
import os
import numpy as np


# path settings
# data_dir = 'D:/hongbin/20231202 CoPtRu real'
icons_dir = 'C:/Users/EEL/PycharmProjects/MOR_RL/software_control/icons'

# cell settings
# ref_potential = -0.263 #for RDE
ref_potential = -0.9 # for Device
counter_potential = 0.9
state_check_potential = 0.65

# protocol settings
protocol_name = {
    'CA': 'C01',
    # 'PEIS': '02_PEIS_C01',
    #'CA_state_check': '02_CA_C01',
}

# EIS settings
circuit = 'R-RQ'
param_df = pd.DataFrame(
    columns=['name', 'value', 'min', 'max'],
    data=[
        ['Rs', 45, 30, 60],
        ['R', 500, 0, 3000],
        ['Q', 1.2e-3, 1e-3, 2e-3],
        ['n', 0.8, 0.7, 1],
    ]
)


# potential conversion vesus ref_potential
def potential_conversion(potential_vs_RHE, ref_potential):
    potential_vs_ref = round(potential_vs_RHE + ref_potential, 3)
    return potential_vs_ref


# convert the action potential
def action_conversion(action, ref_potential):
    action_converted = action.copy()
    action_converted[0] = potential_conversion(action[0], ref_potential)
    action_converted[1] = potential_conversion(action[1], ref_potential)
    action_converted[2] = int(10**action[2])
    action_converted[3] = int(10**action[3])
    print(action_converted)
    return action_converted


def act(step_id, action, data_dir):
    # convert action
    action_converted = action_conversion(action, ref_potential)

    sw = Software(icons_dir)
    act_name = sw.get_act_name(step_id)
    sw.start_exp(act_name, action_converted)
    sw.check_exp_finished()
    # sw.export_to_txt(f'{act_name}_{protocol_name["PEIS"]}.mpr')

    sleep(5)

    # calculate average power output (reward) from two CA steps
    print('action passed', action)
    step_total_time, _ = utils.get_step_actual_time(action)
    if step_total_time > 0:
        ca = Analyzer(os.path.join(data_dir, f"{act_name}_{protocol_name['CA']}.csv"), 'CA')
        ca_total_energy = ca.get_total_energy(ref_potential, counter_potential)

        reward = ca_total_energy / step_total_time
    else:
        reward = 0

    # observe state from EIS
    # eis = Analyzer(f'{data_dir}/{act_name}_{protocol_name["PEIS"]}.mpt', 'EIS')
    # state = eis.get_EIS_fitting_result(circuit, param_df)
    # state_numpy = state.to_numpy()

    # use potential diff / current as state
    state = []
    # ca_state_check = Analyzer(f'{data_dir}/{act_name}_{protocol_name["CA_state_check"]}.csv', 'CA')
    # state.append((counter_potential - state_check_potential) / ca_state_check.data['I/mA'].iloc[0])
    # state.append((counter_potential - state_check_potential) / ca_state_check.data['I/mA'].iloc[-1])

    ca_csv_path = f'{data_dir}/{act_name}_{protocol_name["CA"]}.csv'
    # state_numpy = pd.read_csv(ca_csv_path, sep=';')[['Ewe/V', 'I/mA']].iloc[0:2990].to_numpy().T
    state_numpy = (pd.read_csv(ca_csv_path, sep=';')[['Ewe/V', 'I/mA']].iloc[0:2990].to_numpy()/np.array([1,50])).T
    # state_numpy = np.array(state)

    # reward is a scalar, state is a numpy
    return reward, state_numpy
