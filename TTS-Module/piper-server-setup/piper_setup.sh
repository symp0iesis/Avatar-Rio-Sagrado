#!/bin/bash

if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
apt install python3-venv rubberband-cli
mkdir ~/piper_tts
cd ~/piper_tts
python3 -m venv .venv
source .venv/bin/activate
pip install pip setuptools wheel
git clone https://github.com/rhasspy/piper.git
cd ~/piper_tts/piper/src/python_run
pip install -e .
pip install -r requirements_http.txt
cd ../../..
