# Reduced-Dimensional Fewest-Switches Surface Hopping

This GitHub repository implements the methods outlined in the paper by Morrow, Kwon, Jakubikova, and Kelley (2021). This code has been tested in Anaconda with Python 3.8. The purposes of this repository are to enable reproducibility and to function as a template for researchers wanting to adapt these methods to their own work.

### Requirements
- [Tasmanian](https://tasmanian.ornl.gov/), verified with v7.1
    - Must be added to `PYTHONPATH` post-install, via `export PYTHONPATH=<path-to-Tasmanian>/share/Tasmanian/python:$PYTHONPATH`
- NumPy
- SciPy

### Folders
The `examples` folder provides the codes to generate the initial conditions and run the FSSH swarm. The `src` folder contains the Python drivers: `rrmd_core.py` contains MD and FSSH functions, while all backend mathematical tools are in `rrmd_math_utils.py`.

### Disclaimer
This work represents the opinions and findings of the author(s) and do not represent the official views of the National Science Foundation or the U.S. Government. This code comes with no warranty of any kind, whether expressed or implied.

