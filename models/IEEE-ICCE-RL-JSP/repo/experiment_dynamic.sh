#!/bin/bash

# Start each Python script in the background
python experiment_dynamic_jssp_ieee_icce_rl_jsp.py 0 1 &
python experiment_dynamic_jssp_ieee_icce_rl_jsp.py 2 3 &
python experiment_dynamic_jssp_ieee_icce_rl_jsp.py 4 5 &
python experiment_dynamic_jssp_ieee_icce_rl_jsp.py 6 7 &
python experiment_dynamic_jssp_ieee_icce_rl_jsp.py 8 9 &

# Wait for all to finish
wait
