#!/bin/sh
# launcher.sh

cd ~/Desktop
screen -L -Logfile screenlog.0 -S sagrado-screen
source salto-sagrado/bin/activate
cd Sacred-River-AIvatar/Code/sagrado-loop
python sagrado.py