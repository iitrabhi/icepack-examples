#!/bin/bash
# Activate Firedrake
source /home/firedrake/.bashrc
source /home/firedrake/firedrake/bin/activate
# Start Jupyter Lab
jupyter lab --ip 0.0.0.0 --no-browser --allow-root 
