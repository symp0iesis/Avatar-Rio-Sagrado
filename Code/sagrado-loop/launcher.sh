#!/bin/bash
# launcher.sh

cd ~/Desktop
screen -L -Logfile screenlog.0 -dm -S sagrado-screen bash -c "source salto-sagrado/bin/activate && cd Sacred-River-AIvatar/Code/sagrado-loop && python sagrado.py"

# source salto-sagrado/bin/activate
# cd Sacred-River-AIvatar/Code/sagrado-loop
# python sagrado.py