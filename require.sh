#!/bin/bash
sudo apt-get upgrade
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
  libsndfile-dev portaudio19-dev

sudo apt install qt5-default
###
#python3 -m pip install virtualenv
#
#virtualenv venv
#source ./venv/bin/activate

pip install -r requirements.txt

git clone git@lab.hanium.or.kr:21_HF433/21_hf433-front.git

cd 21_hf433-front

pyinstaller --onefile ./main.py

cp -r ./assets ./dist/assets
cp -r ./domain ./dist/domain
cp -r ./ui ./dist/ui
cp -r ./capstone*.json ./dist
cd ./dist
