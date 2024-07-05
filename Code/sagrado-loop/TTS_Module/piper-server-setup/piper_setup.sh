#!/bin/bash

apt install python3-venv rubberband-cli
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

#Change permissions so Piper model can be loaded from within Python code without sudo.
chmod 777 ~/piper_tts/
chmod 777 ~/piper_tts/pt_BR-faber-medium.onnx
chmod 777 ~/piper_tts/pt_BR-faber-medium.onnx.json
