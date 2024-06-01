from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch


class STTEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # load model and processor for speech-to-text in Portuguese, based on Wav2Vec2
        self.processor_stt = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-portuguese")
        self.model_stt = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-portuguese")
        torch.set_grad_enabled(False)
        self.model_stt.eval()
        # torch.set_num_threads(4)
        
    def speech_to_text(self, speech_data, sampling_rate):
        with torch.no_grad():
            input_features = self.processor_stt(speech_data, sampling_rate=sampling_rate, return_tensors="pt", padding="longest")
            logits = self.model_stt(input_features.input_values, attention_mask=input_features.attention_mask).logits
            predicted_id = torch.argmax(logits, dim=-1)
            text = self.processor_stt.batch_decode(predicted_id)[0]
            return text
