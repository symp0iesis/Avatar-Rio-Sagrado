#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -m MODEL_NAME" >&2
    exit 1
}

# Parse options
while getopts ":m:" opt; do
    case $opt in
        m)
            MODEL="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Check if MODEL variable is set
if [ -z "$MODEL" ]; then
    echo "Model name not provided."
    usage
fi

cd ~/piper_tts
source .venv/bin/activate
python3 -m piper.http_server --model "$MODEL"
