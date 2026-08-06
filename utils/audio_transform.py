import torch
import torchaudio


class AudioTransform(torch.nn.Module):

    
    def __init__(
        self,
        sample_rate=44100,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    ):

        super().__init__()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        self.db = torchaudio.transforms.AmplitudeToDB()


    def forward(self, audio):

        mel = self.mel(audio)

        mel = self.db(mel)

        mel = (
            mel - mel.mean()
        ) / (
            mel.std() + 1e-8
        )

        return mel
