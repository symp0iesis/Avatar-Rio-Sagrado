# TTS Module for Raspberry Pi

In this folder, there are 2 options for text-to-speech, each of them coming with a demo web-app written using streamlit. Both modules break the text in sentences, send each sentence to the TTS engine one by one, then wait, in a separate thread, for the prediction from the TTS engine and "speak" the result after receiving it.

## 1. Coqui TTS

For this method, we use Coqui TTS module, capable to clone a voice and speak the text given in the prompt, in a language selected by the user

## 2. Piper TTS

This method uses Piper TTS, which is a TTS module optimized for Rasperry Pi 4. In order to use this, we need to setup the Piper server using the script *piper_setup.sh* from the folder *piper-server-setup/* and run the server using the command  *piper_server.sh -m `<model name>`* from the same folder. Since the TTS model is used for text in Portuguese, the used model name in this case will be *pt_BR-faber-medium* which will automatically be downloaded. Other speakers, languages and models can be found [here](https://huggingface.co/rhasspy/piper-voices/tree/main).
