import random
import torch
import torch.nn.functional as F
import librosa


class AudioAugmentation:
    """
    Clase para aplicar técnicas de Data Augmentation a audios.
    """

    def __init__(
        self,
        noise_factor=0.005,
        shift_max=0.2,
        verbose=True
    ):
        self.noise_factor = noise_factor
        self.shift_max = shift_max
        self.verbose = verbose

    def add_noise(self, audio):
        """
        Agrega ruido gaussiano al audio.
        """
        noise = torch.randn_like(audio)
        return audio + self.noise_factor * noise

    def time_shift(self, audio):
        """
        Desplaza el audio hacia adelante o atrás.
        """
        shift = int(
            audio.shape[-1] *
            random.uniform(-self.shift_max, self.shift_max)
        )

        return torch.roll(audio, shifts=shift, dims=-1)

    def change_volume(self, audio, gain_min=0.7, gain_max=1.3):
        """
        Cambia el volumen del audio aplicando una ganancia aleatoria.
        """

        gain = random.uniform(gain_min, gain_max)

        if self.verbose:
            print(f"Ganancia aplicada: {gain:.2f}")

        augmented = audio * gain

        # Evitar saturación
        augmented = torch.clamp(augmented, -1.0, 1.0)

        return augmented

    def pitch_shift(self, audio, sr, n_steps=(-2, 2)):
        """
        Cambia el tono del audio sin modificar su duración.
        """

        audio_np = audio.squeeze(0).cpu().numpy()

        steps = random.uniform(n_steps[0], n_steps[1])

        if self.verbose:
            print(f"Cambio de tono: {steps:.2f} semitonos")

        audio_shifted = librosa.effects.pitch_shift(
            y=audio_np,
            sr=sr,
            n_steps=steps
        )

        return torch.tensor(
            audio_shifted,
            dtype=torch.float32
        ).unsqueeze(0)

    def time_stretch(self, audio, rate_min=0.9, rate_max=1.1):
        """
        Cambia la velocidad del audio sin modificar el tono.
        Mantiene el mismo número de muestras.
        """

        original_length = audio.shape[-1]

        audio_np = audio.squeeze(0).cpu().numpy()

        rate = random.uniform(rate_min, rate_max)

        if self.verbose:
            print(f"Velocidad aplicada: {rate:.2f}x")

        stretched = librosa.effects.time_stretch(
            y=audio_np,
            rate=rate
        )

        stretched = torch.tensor(
            stretched,
            dtype=torch.float32
        )

        current_length = stretched.shape[0]

        # Si el audio quedó más largo
        if current_length > original_length:
            stretched = stretched[:original_length]

        # Si quedó más corto
        elif current_length < original_length:
            padding = original_length - current_length
            stretched = F.pad(stretched, (0, padding))

        return stretched.unsqueeze(0)

    def __call__(self, audio, sr):
        """
        Aplica aumentaciones aleatorias.
        """

        augmentations = []

        # Ruido
        if random.random() < 0.5:
            audio = self.add_noise(audio)
            augmentations.append("Ruido")

        # Desplazamiento temporal
        if random.random() < 0.5:
            audio = self.time_shift(audio)
            augmentations.append("Desplazamiento temporal")

        # Cambio de volumen
        if random.random() < 0.5:
            audio = self.change_volume(audio)
            augmentations.append("Cambio de volumen")

        # Cambio de tono
        if random.random() < 0.5:
            audio = self.pitch_shift(audio, sr)
            augmentations.append("Pitch Shifting")

        # Cambio de velocidad
        if random.random() < 0.5:
            audio = self.time_stretch(audio)
            augmentations.append("Time Stretch")

        # Limitar amplitud final
        audio = torch.clamp(audio, -1.0, 1.0)

        if self.verbose:
            if not augmentations:
                augmentations.append("Sin augmentación")

            print("Aumentaciones aplicadas:", ", ".join(augmentations))

        return audio