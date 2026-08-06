import random
import torchaudio


class SpecAugmentation:
    """
    Aplica aumentaciones sobre el espectrograma Mel.
    """

    def __init__(self):

        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=30
        )

        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=10
        )

    def __call__(self, mel):

        techniques = []

        if random.random() < 0.5:
            mel = self.time_mask(mel)
            techniques.append("Time Mask")

        if random.random() < 0.5:
            mel = self.freq_mask(mel)
            techniques.append("Frequency Mask")

        if not techniques:
            techniques.append("Sin SpecAugment")

        print("SpecAugment:", ", ".join(techniques))

        return mel