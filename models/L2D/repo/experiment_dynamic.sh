#!/bin/bash

# Start each Python script in the background
python experiment_dynamic_jssp_l2d.py SavedNetwork/15_15_1_99.pth &
python experiment_dynamic_jssp_l2d.py SavedNetwork/30_15_1_199.pth &
python experiment_dynamic_jssp_l2d.py SavedNetwork/30_20_1_199.pth &
python experiment_dynamic_jssp_l2d.py SavedNetwork/6_6_1_99.pth &
python experiment_dynamic_jssp_l2d.py SavedNetwork/20_20_1_199.pth &
python experiment_dynamic_jssp_l2d.py SavedNetwork/30_20_1_99.pth &
python experiment_dynamic_jssp_l2d.py SavedNetwork/20_15_1_199.pth &
python experiment_dynamic_jssp_l2d.py SavedNetwork/30_15_1_99.pth &
python experiment_dynamic_jssp_l2d.py SavedNetwork/10_10_1_99.pth &
python experiment_dynamic_jssp_l2d.py SavedNetwork/20_20_1_99.pth &
python experiment_dynamic_jssp_l2d.py SavedNetwork/20_15_1_99.pth &

# Wait for all to finish
wait
