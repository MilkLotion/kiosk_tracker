#!/bin/bash
sudo sh -c 'echo 100 > /sys/devices/pwm-fan/target_pwm'
sudo apt-get upgrade
sudo apt-get update
pip install -r requirements.txt
# client system execute
python3 main.py;