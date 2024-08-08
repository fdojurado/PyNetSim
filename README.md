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

```python
from pynetsim import PyNetSim

config_file = "config.yaml"

# Create a PyNetSim instance
pynetsim = PyNetSim(config_file)

# Run the simulation
pynetsim.run()
```

## Documentation

For more information on how to use PyNetSim, please refer to the [documentation](https://pynetsim.readthedocs.io/).

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
