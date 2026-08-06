import torch
import torchaudio


class AudioTransform:
    """
    Convierte un audio en un Mel Spectrogram.
    """

    def __init__(
        self,
        sample_rate=44100,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    ):

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        self.db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, audio):

        mel = self.mel(audio)

        mel = self.db(mel)

        return mel