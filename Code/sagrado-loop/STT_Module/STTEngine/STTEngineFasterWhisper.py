from faster_whisper import WhisperModel
import torch


class STTEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.model_size = "tiny"
        self.model_stt = WhisperModel(self.model_size, local_files_only=True)
        torch.set_grad_enabled(False)
        # torch.set_num_threads(4)

    def speech_to_text(self, speech_data, sampling_rate):
        with torch.no_grad():
            segments, _ = self.model_stt.transcribe(speech_data, language="pt")
            text = ''.join(segment.text for segment in segments)
            return text
