<!-- <img src="https://github.com/SDWSN-controller/SDWSN-controller.github.io/blob/develop/images/logo/Contiki_logo_2RGB.png" alt="Logo" width="256"> -->

# PyNetSim: Wireless Sensor Network Simulator

**Warning:** This documentation is currently under construction. Some information may be incomplete or subject to change.

PyNetSim stands out as a network simulator specifically crafted for the evaluation of LEACH-based protocols, including traditional LEACH and its variants like LEACH-C. Developed in Python, this tool provides a versatile and accessible platform for researchers and students to test, analyze, and refine their clustering algorithms in the context of wireless sensor networks.

Key Features:

- LEACH-Based Protocol Testing: PyNetSim is purpose-built for assessing LEACH, LEACH-C, and other related protocols, enabling users to conduct comprehensive evaluations and comparisons.
- Python-Powered: Leveraging the simplicity and versatility of Python, PyNetSim offers an environment conducive to seamless experimentation and adaptation. Researchers and students can easily integrate and modify components to tailor simulations to their specific needs.

Whether you are a researcher aiming to optimize LEACH-based protocols or an educator guiding students through the complexities of wireless sensor networks, PyNetSim provides a reliable and efficient platform for your exploration and experimentation needs.

## Installation

To install PyNetSim, you can use the following pip command:

```bash
pip install pynetsim
```

## Usage

To use PyNetSim, you can follow the example below:

### Create a Configuration File

Create a configuration file in YAML format with the following structure:

```yaml
name: example network
seed: 1234
# Set the save path for the results
save_path: /Users/fernando/PyNetSim/tutorials/results/leach/
network:
  plot: False
  num_sensor: 100
  transmission_range: 60
  model: extended
  protocol:
    name: LEACH
    init_energy: 0.5
    rounds: 8000
  width: 100
  height: 100
  num_sink: 1
```

### Run the Simulation

```python
from pynetsim import PyNetSim

# Set the path to the configuration file
config_file = "config.yaml"

# Create a PyNetSim instance
pynetsim = PyNetSim(config_file)

# Run the simulation
pynetsim.run()
```

A new folder will be created with the results of the simulation.

To plot the results, you can use the following code:

```python
py plot_results.py -i /path/to/results/ -o /path/to/output/
```

<!-- Section that explain how to run the experiments in the paper -->
## Experiments in the Paper

To reproduce the experiments in the paper, you can use refer to the folder [experiments](tutorials/experiments). The folder contains the configuration files used in the experiments, as well as the scripts to run the simulations.

### Running LEACH

For example, to run five simulations of LEACH using the paper topology, you can use the following command:

```bash

py mult_sim_seed.py -c ./experiments/leach.yml -n leach -r 5

```

### Running LEACH-C

To run five simulations of LEACH-C using the paper topology, you can use the following command:

```bash
py mult_sim_seed.py -c ./experiments/leach-c.yml -n leach-c -r 5
```

### Running LEACH-RLC

Before running LEACH-RLC, you need to download the trained model from the following link: [Models](https://zenodo.org/records/13253417/files/models.zip?download=1&preview=1).

Now, you need to extract the models.zip file and replace in the [LEACH-RLC yaml](tutorials/experiments/leach-rlc.yml) file the following fields:

```yaml
# Path to the cluster head model
cluster_head_model: /path/to/cluster_heads/ch_model.pt
# Cluster head path to the data, useful for mean and std
cluster_head_data: /path/to/cluster_heads/data/data.csv
# Path to the cluster assignment model
cluster_assignment_model: /path/to/cluster_assignment/cluster_assignment_model.pt
# Cluster assignment path to the data, useful for mean and std
cluster_assignment_data: /path/to/cluster_assignment/data/data.csv
```


To run five simulations of LEACH-RLC using the paper topology, you can run the below command within the tutorials folder:

```bash
py rl/mult_sim_seed.py -c ./experiments/leach-rlc.yml -n leach-rlc -m /path/to/rl-agent/rl_model.zip -l dqn_leach_add_log/ -r 5
```

### Running EE-LEACH

To run five simulations of EE-LEACH using the paper topology, you can use the following command:

```bash
py mult_sim_seed.py -c ./experiments/ee-leach.yml -n ee-leach -r 5
```

### Running LEACH-D

To run five simulations of LEACH-D using the paper topology, you can use the following command:

```bash
py mult_sim_seed.py -c ./experiments/leach-d.yml -n leach-d -r 5
```

### Running LEACH-CM

To run five simulations of LEACH-CM using the paper topology, you can use the following command:

```bash
py mult_sim_seed.py -c ./experiments/leach-cm.yml -n leach-cm -r 5
```


### Plotting the Results

We now have the results of the simulations in individual folders. To plot the results, you can use the following command:

```bash
py plot_results.py -i /path/to/results/leach /path/to/results/leach-c -o /path/to/output/
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PyNetSim in your research, please cite the following paper:

```
@article{pynetsim,
  title={LEACH-RLC: Enhancing IoT Data Transmission with Optimized Clustering and Reinforcement Learning},
  author={Jurado-Lasso, F Fernando and Jurado, JF and Fafoutis, Xenofon},
  journal={IEEE Internet of Things Journal},
  year={2024},
  publisher={IEEE}
}
```
