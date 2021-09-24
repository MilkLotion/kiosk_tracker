#!/bin/bash
sudo sh -c 'echo 100 > /sys/devices/pwm-fan/target_pwm'
# client system execute
python3 main.py;