#!/bin/bash

# Start each Python script in the background
python experiment_dynamic_jssp_l2d.py 0 &
python experiment_dynamic_jssp_l2d.py 1 &
python experiment_dynamic_jssp_l2d.py 2 &
python experiment_dynamic_jssp_l2d.py 3 &
python experiment_dynamic_jssp_l2d.py 4 &
python experiment_dynamic_jssp_l2d.py 5 &
python experiment_dynamic_jssp_l2d.py 6 &
python experiment_dynamic_jssp_l2d.py 7 &

# Wait for all to finish
wait
