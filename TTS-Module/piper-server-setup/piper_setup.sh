#!/bin/bash

# Create virtual environment and install dependencies
mkdir ~/piper_tts
cd ~/piper_tts
python3 -m venv .venv
source .venv/bin/activate
pip install pip setuptools wheel
git clone https://github.com/rhasspy/piper.git
cd piper/src/python_run
pip install -e .
pip install -r requirements_http.txt
cd ../../..
