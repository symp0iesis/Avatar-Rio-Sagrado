import os, threading, requests, sounddevice
from time import time, sleep
import numpy as np
from numpy import array

#To toggle modifications needed to make code work on Mayowa's computer
on_mac = True


# if on_mac == True:
#     os.environ['KMP_DUPLICATE_LIB_OK']='True'


def init_stt():
    print('Initializing Speech-to-Text...')
    #Set up STT
    import torch
    import pyaudio
    from STT_Module.STTEngine.STTEngineFasterWhisper import STTEngine
    global stream, VAD_WINDOW_LENGTH, vad_iterator, speech_data, start_speech, end_speech, stt_engine, SAMPLING_RATE

    SAMPLING_RATE = 16000
    VAD_WINDOW_LENGTH = 512 #1600
    VAD_THRESHOLD = 0.4
    MIN_SILENCE_DURATION_MS = 500
    SPEECH_PAD_MS = 100

    # load model and processor for speech-to-text
    stt_engine = STTEngine()

    # load model and utils for voice activity detection
    model_vad, utils_vad = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=False)
    (_, _, _, VADIterator, _) = utils_vad
    vad_iterator = VADIterator(model_vad, threshold=VAD_THRESHOLD, min_silence_duration_ms=MIN_SILENCE_DURATION_MS, speech_pad_ms=SPEECH_PAD_MS)

    # initialize listening device
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLING_RATE, input=True, frames_per_buffer=VAD_WINDOW_LENGTH)
    speech_data = np.array([])
    start_speech = False
    end_speech = False

    print('Done.\n')

init_stt()


#Set up TTS
from soundfile import write as sf_write
from sounddevice import play, wait
import nltk.data
print(' Loading nltk tokenizer...')
os.environ['NLTK_DATA'] = 'TTS_Module/nltk_data/'
sentence_tokenizer = nltk.data.load('tokenizers/punkt/PY3/portuguese.pickle')
# nltk.download('punkt')
print(' Done.')

def init_coqui_tts():
    print('Initializing Text-to-Speech...')
    from torch.cuda import is_available as cuda_is_available
    from TTS.api import TTS
    global split_text_into_sentences, text_to_speech

    def load_tts_model(device):
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


    def split_text_into_sentences(text):
        sentences = sentence_tokenizer.tokenize(text)
        # sentences = nltk.sent_tokenize(text)
        for i in range(len(sentences)):
            if sentences[i][-1] == '.':
                sentences[i] = sentences[i][:-1]
        return sentences

    device = "cuda" if cuda_is_available() else "cpu"
    # device = "cpu"
    tts = load_tts_model(device)

    def text_to_speech(sentence):
        wav_data = tts.tts(text=sentence, language='pt', speaker_wav="TTS_Module/cr7.wav", split_sentences=False)
        return wav_data

def init_piper_tts(): 
    print('Initializing Text-to-Speech...')
    import pyrubberband
    global split_text_into_sentences, text_to_speech
    HEADER_SIZE = 44
    SAMPLE_RATE = 21000
    SPEECH_SPEED = 0.835


    def split_text_into_sentences(text):
        sentences = sentence_tokenizer.tokenize(text)
        # sentences = nltk.sent_tokenize(text)
        return sentences

    def bytes_to_sound(data):
        sound_data = np.frombuffer(data, dtype=np.int16, count=len(data) // 2).astype(np.float64) / 32768.0
        return sound_data

    def text_to_speech(text):
        URL = "http://localhost:5000"
        resp = requests.get(URL, params={'text': text})
        if resp.status_code == 200:
            wav_data = bytes_to_sound(resp.content[HEADER_SIZE:])
            if wav_data is not None:
                slowed_wav_data = pyrubberband.time_stretch(wav_data, SAMPLE_RATE, SPEECH_SPEED)
                return slowed_wav_data
            else:
                raise Exception("Failed to convert text to speech")
        return None

    def startup_piper_server():
        print('---------\n Starting up Piper Server...')
        # import subprocess
        # subprocess.run("cat TTS_Module/piper-server-setup/piper_server.sh; sh TTS_Module/piper-server-setup/piper_server.sh -m pt_BR-faber-medium >> piper_server_output.txt 2>&1", shell=True) #&>> piper_server_output.txt
        os.system("sh TTS_Module/piper-server-setup/piper_server.sh -m pt_BR-faber-medium >> piper_server_output.txt 2>&1") #&>> piper_server_output.txt
        print(' Done.\n---------')

    #Start Piper webserver in a thread
    piper_thread = threading.Thread(target=startup_piper_server)
    piper_thread.start()



# Select TTS Backend to Use:
# init_coqui_tts()
init_piper_tts()

def play_audio():
    print('play_audio() called')
    global audio_queue
    while True:
        sleep(0.3)
        if len(audio_queue) > 0:
            wav_data = audio_queue.pop(0)
            # print('Playing generated audio..', type(wav_data))
            play(array(wav_data), 24000)
            wait()



# Global audio queue to store the audio data
audio_queue = []
file_data = []

# Start the audio playback thread
audio_thread = threading.Thread(target=play_audio)
# audio_thread.daemon = True
audio_thread.start()


def speak(input_text):
    sentences = split_text_into_sentences(input_text)
    for sentence in sentences:
        wav_data = text_to_speech(sentence)
        if wav_data is None:
            raise Exception("Piper Server Error: Error generating speech from Piper TTS.")
        # print('Generated audio data')
        audio_queue.append(wav_data)
        file_data.extend(wav_data)
        # print('Audio array length: ', len(audio_queue))

    print('Writing audio to file...')
    sf_write("output.wav", array(file_data), 24000)
    # os.remove("output.wav")


print('Done.\n')


# # To just test the TTS
# avatar_response = 'Olá, eu estou sempre fluindo, sempre renovando-me nas águas que percorrem o município de Morretes até a baía de Antonina. Como posso ajudar você hoje?'
# speak(avatar_response)


def init_retune():
    print('Setting up Retune communication utils...')
    #Utilities for Communicating with Retune Avatar

    global json_data, generate_response_url, headers

    create_thread_url = 'https://retune.so/api/chat/11ee2b4b-d054-47f0-9771-310dd2eca1c4/new-thread'
    generate_response_url = 'https://retune.so/api/chat/11ee2b4b-d054-47f0-9771-310dd2eca1c4/response'


    headers = {
        'Content-Type': 'application/json',
        'X-Workspace-API-Key': '11ee2f36-c329-7e80-9c79-ff4ddae14db0',
    }


    #Create new conversation thread
    print('\n Creating new conversation thread with Retune avatar...')
    response = requests.get(
        create_thread_url,
        headers=headers,
    )
    print(' Done.\n')

    #Could implement simultaneous conversations with multiple threads at some point,
    # depending on how this is planned to be deployed.
    # print('Response: ', response.json())
    threadId = response.json()['threadId']

    json_data = {
        'threadId': threadId, #'11ef1c61-8795-5150-af65-e1c12ce1ce56',
        'input': 'Hello! Please speak English to me. What are you about?',
    }


    print('Done.\n')

init_retune()

def avatar_response(speech_text):
    # print('Getting avatar response...')
    json_data['input'] = speech_text
    response = requests.post(
        generate_response_url,
        headers=headers,
        json=json_data,
    )

    #Issue communicating with Retune, and obtaining avatar response
    if response.status_code != 200:
        avatar_response = "Retune Error." #In Portuguese? What would Danilo want?
    else:
        avatar_response = response.json()['response']['value']

    print('Avatar response: ', avatar_response)

    speak(avatar_response)

#Main loop. Listens for audio, transcribes it, passes it to Retune Avatar, 
#converts Avatar output text to speech.

# print("Listening for speech...")

while True:
    if on_mac == True:
        data_np = np.frombuffer(stream.read(VAD_WINDOW_LENGTH, exception_on_overflow = False), dtype=np.float32)
    else:
        data_np = np.frombuffer(stream.read(VAD_WINDOW_LENGTH), dtype=np.float32)
    speech_dict = vad_iterator(data_np, return_seconds=True)
    if speech_dict:
        print('Speech dict: ', speech_dict)
        if 'start' in speech_dict:
            start_speech = True
            end_speech = False
        elif 'end' in speech_dict:
            end_speech = True
            start_speech = False
    if start_speech and not end_speech:
        speech_data = np.append(speech_data, data_np)        
    elif end_speech and not start_speech:
        speech_data = np.append(speech_data, data_np)
        # start_time = time()
        speech_text = stt_engine.speech_to_text(speech_data, SAMPLING_RATE)
        # end_time = time()
        print('Transcription: ', speech_text)
        # print(f"Time taken: {end_time - start_time:.2f} seconds\n")
        speech_data = np.array([])
        start_speech = False
        end_speech = False

        # print('Getting avatar response...')
        # speech_text = input('Enter input: ')

        avatar_response(speech_text)

