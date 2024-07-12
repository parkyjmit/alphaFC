# $\alpha$ Fuel Cell
This is the codebase of the paper titled **Alpha-Fuel-Cell: Maximizing Delivered Power for Direct Methanol Fuel Cell with Actor-Critic Algorithm**.

## Abstract
Hongbin Xu $^{1\dagger}$, Yang Jeong Park1 $^{2\dagger}$. Zhichu Ren $^{1}$, Daniel J. Zheng $^{1}$, Haojun Jia $^{2}$, Chenru Duan $^{2}$, Guanzhou Zhu $^{1}$, Yuriy Román-Leshkov $^{2}$, Yang Shao-Horn $^{1,3*}$, Ju Li $^{1,4*}$   

1 Department of Materials Science and Engineering, Massachusetts Institute of Technology; Cambridge, MA 02139, USA   
2 Department of Chemical Engineering, Massachusetts Institute of Technology; Cambridge, MA 02139, USA   
3 Department of Mechanical Engineering, Massachusetts Institute of Technology; Cambridge, MA 02139, USA   
4 Department of Nuclear Science and Engineering, Massachusetts Institute of Technology; Cambridge, MA 02139, USA   

† These authors contributed equally: Hongbin Xu, Yang Jeong Park

*Corresponding author. Email: shaohorn@mit.edu (Yang Shao-Horn); liju@mit.edu (Ju Li)

Taking time-dependent actions to tune the performance of complex energy systems is desirable. Fuel cell should economically delivered the maximum power over a long time period, but as the electrocatalytic surfaces become fouled, theirits performance decays over time. Changing the potential dynamically during operation can clean the surface and recover the activity of catalysts for direct methanol fuel cells (DMFCs). However, manual experiments and parameter adjustments face many drawbacks. Here we developed and demonstrated a nonlinear policy model (Alpha-Fuel-Cell) inspired by actor-critic reinforcement learning (RL) to control and maximize the time-averaged delivered power for DMFCs. Our policy model learns directly from real-world electrical current trajectories to infer the state of catalysts during operation. Combineding with the action parameters, the cell power can be predicted and a suitable action for the next step is generated automatically, which can be utilized to further maximize the delivered power in the period. Moreover, the model can provide protocols to achieve the required power while significantly slowing the degradation of catalysts significantly. Benefiting from this model, the time-averaged power delivered is 285.2% and 153.8% compared to steady operation for three-electrode cells and DMFCs in over 12 hours, respectively. Our framework can be generalized to other energy applications requiring long time horizon decision-making in the real world. 

# Fuel Cell Control and Neural Network Training

This repository contains two main components for controlling a DMFC system using neural networks:

1. **Fuel Cell Operation Control**: The code operates the fuel cell and adjusts the output power based on the control signals from a neural network.
2. **Neural Network Training**: The code trains the neural network using data obtained from the fuel cell operation.

## 1. Environment
The exported environment is provided.
```
conda env create -f environment.yaml
conda activate alphaFC
```
## 2. How to run
1. Determine saving folder
   - `config.yaml` change line 6
2. Copy and paste following files in the directory
   - `running_actor.pt`
   - `running_model.pt`
   - `target.json`
3. If you want to change the strategy
    - go to `agents.py` line 55,56
    - if both commented, then our model
4. type the code in the terminal`python .\ml\conduct_experiment.py`


## Components

### 1. Fuel Cell Operation Control
- **Description**: This module controls the fuel cell to obtain the desired output power.
- **Functionality**: 
  - Uses a neural network to control the output power.
  - If a trained neural network is available, it will be used to control the fuel cell.
  - If no trained neural network is available, a randomly initialized neural network will be used.
- **Files**: 
  - `config.yaml`: Determine configuration of operation. 
  - `target.json`: 
  - `fuel_cell_control.py`: Main script for controlling the fuel cell.
  - `running_actor.pt`
  - `running_model.pt`

### 2. Neural Network Training
- **Description**: This module trains the neural network using data obtained from the fuel cell operation.
- **Functionality**:
  - Collects data from the fuel cell operation.
  - Trains the neural network using the collected data.
- **Files**:
  - `config.yaml`: Determine configuration of operation.
  - `neural_network_training.py`: Main script for training the neural network.

## Workflow

1. **Data Collection and Training**:
   - Run the `neural_network_training.py` script to collect data from the fuel cell and train the neural network.
   - The script saves the trained neural network model for later use.

2. **Fuel Cell Control**:
   - Run the `fuel_cell_control.py` script to operate the fuel cell.
   - The script checks for the presence of a trained neural network model.
   - If a trained model is found, it will be used for controlling the fuel cell.
   - If no trained model is found, a randomly initialized neural network will be used instead.

## Getting Started

### Prerequisites
- Python 3.x
- Required Python libraries (see `requirements.txt` for details)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fuel-cell-control.git
    cd fuel-cell-control
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Train the Neural Network**:
    ```bash
    python neural_network_training.py
    ```

2. **Control the Fuel Cell**:
    ```bash
    python fuel_cell_control.py
    ```

## Contributing

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions, feel free to reach out to us at [your-email@example.com](mailto:your-email@example.com).

