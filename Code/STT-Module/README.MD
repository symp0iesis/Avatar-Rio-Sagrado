# STT Module for Raspberry Pi

In this folder, we have the speech-to-text module of the Sacred River AIvatar. In order to run it, use script *stt.py*, where we use a voice activity detector to capture speech from microphone, and then transcribe the speech using one of the two speech-to-text models available on the Hugging Face platform. As it is mentioned before, the app consists of two main parts:

## 1. Voice Activity Detector (VAD)

Immediately after capturing the sound from the microphone, with the use of a VAD, we recognize the portions that contain actual human speech, portions that will be send to the speech-to-text model for transcription. The VAD used in this case is [Silero VAD](https://github.com/snakers4/silero-vad), a pretrained, lightweight, enterprise-grade voice activity detector, perfect for Raspberry Pi.

## 2. Speech-to-Text Model (STT)

After we obtain the speech portions, we pass them to the STT model, which will transcribe the speech into text. Since our expected speech is in Portuguese, we have prepared two possible models for STT:

### a) Wav2Vec2-Large XLSR-53 for Portuguese

Found [here](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-portuguese), the model is a Wav2Vec2-Large XLSR-53 fine-tuned on Portuguese. The model gives very good results in a quite short time, since it is specialized on the Portuguese language.

### b) OpenAI's Whisper Tiny

Found [here](https://huggingface.co/openai/whisper-tiny), this model is the tiny version of the model Whisper, a multilingual, automatic speech recognition model developed by OpenAI, capable of recognizing speech from 99 language, including Portuguese. Since it is the tiny version of the model, the inference is very fast and surprisingly accurate, making it an another perfect fit for Raspberry Pi.


## Notes on Requirements

In order to have the module working on a raspberry pi, a 64bit OS is required. Once that's done, make sure to sudo apt install portaudio (so you can then later pip install pyaudio). Newer (2023 onwards) OS versions of the pi won't allow direct pip3 install requets so it's important to create a virtual environment to get things working. Once that's setup, you should be able to install the requirements.