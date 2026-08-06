import torch.nn as nn
import torchaudio


class AudioTransform(nn.Module):
   

    def __init__(self):

        super().__init__()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )

        self.db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, audio):

        # Mover las transformaciones al mismo dispositivo
        device = audio.device

        self.mel = self.mel.to(device)
        self.db = self.db.to(device)

        mel = self.mel(audio)
        mel = self.db(mel)

        return mel
