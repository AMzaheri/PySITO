
#----------------------------------PySITo
A Python tool for Seismic Imaging wavefield Tomography

Overview
PySITo is a seismic imaging tool designed for wavefield tomography, utilizing Devito for efficient seismic wave simulation. This package aims to support Carbon Capture and Storage (CCS) site monitoring by reconstructing subsurface velocity changes.

Current Work
Issue: The gradient calculations are not currently producing correct results..
Next Steps: After debugging the gradients, the package will be parallelised for better performance using MPI.

Installation (Temporary)
An  environment file will be provided to install Devito and other required dependencies.

Usage
1- Set input parameters in inputs/inputs_idwt.ini.
2- Run the inversion by executing:
    python run_idwt.py

Planned Improvements
1- Gradient Debugging: Fixing issues with gradient calculations.
2- Implement parallelism using MPI for larger simulations.
3- Provide a YAML file for easy dependency installation.
