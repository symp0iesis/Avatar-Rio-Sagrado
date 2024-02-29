from soundfile import write as sf_write
import threading
import numpy as np
import requests
from time import sleep
from sounddevice import play, wait
import pyrubberband
import nltk
nltk.download('punkt')

HEADER_SIZE = 44
SAMPLE_RATE = 21000
SPEECH_SPEED = 0.835
URL = "http://localhost:5000"

def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def bytes_to_sound(data):
    sound_data = np.frombuffer(data, dtype=np.int16, count=len(data) // 2).astype(np.float64) / 32768.0
    return sound_data

def text_to_speech(url, text):
    resp = requests.get(url, params={'text': text})
    if resp.status_code == 200:
        return bytes_to_sound(resp.content[HEADER_SIZE:])
    return None

def play_audio():
    global audio_queue
    while True:
        sleep(0.5)
        if len(audio_queue) > 0:
            wav_data = audio_queue.pop(0)
            play(np.array(wav_data), SAMPLE_RATE)
            wait()

# Global audio queue to store the audio data
audio_queue = []
file_data = []

# Start the audio playback thread
audio_thread = threading.Thread(target=play_audio)
audio_thread.daemon = True
audio_thread.start()

while True:
    input_text = input('Enter the text you want to convert to speech in Portuguese (or type "exit" to quit): ')
    if input_text.lower() == "exit":
        break

    if input_text:
        sentences = split_text_into_sentences(input_text)
        for sentence in sentences:
            print(sentence)
            wav_data = text_to_speech(URL, sentence)
            if wav_data is not None:
                slowed_wav_data = pyrubberband.time_stretch(wav_data, SAMPLE_RATE, SPEECH_SPEED)
                audio_queue.append(slowed_wav_data)
                file_data.extend(slowed_wav_data)
            else:
                print("Failed to convert text to speech")
                raise Exception("Failed to convert text to speech")

        sf_write("output.wav", np.array(file_data), SAMPLE_RATE)
        print("File output.wav created")
        # Clear file_data after writing to allow for new inputs
        file_data.clear()
