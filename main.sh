#!/bin/bash
sudo sh -c 'echo 100 > /sys/devices/pwm-fan/target_pwm'
./21_hf433-front/main
python3 main_tracker.py;