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

## 1. Install
1. Clone the repository:
    ```bash
    git clone https://github.com/parkyjmit/alphaFC.git
    cd alphaFC
    ```
2. The exported environment is provided. This work is optimized for Windows OS, since our work utilizes EC-Lab software, mainly works on Windows.
    ```bash
    conda env create -f environment.yaml
    conda activate alphaFC
    ```
## 2. Components

### 1. Fuel Cell Operation Control
- **Description**: This module controls the fuel cell to obtain the desired output power.
- **Functionality**: 
  - Uses a neural network to control the output power.
  - The script checks for the presence of a trained neural network model.
    - If a trained neural network is available, it will be used to control the fuel cell.
    - If no trained neural network is available, a randomly initialized neural network will be used.
- **Files**: 
  - `config.yaml`: Determine configuration of operation. 
  - `target.json`: To control the output power of the DMFC. It can be modified during the cell operation.
  - `fuel_cell_control.py`: Main script for operation of the fuel cell.
  - `running_actor.pt`: The trained neural network which plays a role as an actor of the dynamic control system. If it doesn't exist, the algorithm define randomly initialized neural network.
  - `real_database.json` or `sim_database.json`: The cell operation is recorded in this file.
- **How to run**:
  - Run `python .\ml\fuel_cell_control_experiment.py` to operate the DMFC.

### 2. Neural Network Training
- **Description**: This module trains the neural network using data obtained from the fuel cell operation.
- **Functionality**:
  - Trains the critic neural network module using the collected data.
  - The script saves the trained neural network model for later use.
- **Files**:
  - `config.yaml`: Determine configuration of training.
  - `neural_network_training.py`: Main script for training the neural network. 
  - `running_model.pt`: The neural network on training phase. 
- **How to run**:
  - Run `python .\ml\neural_network_training.py` to train the critic neural network module from collected data.

## License

This project is licensed under the MIT License. 

# Cite this work
Citation information will be provided soon.