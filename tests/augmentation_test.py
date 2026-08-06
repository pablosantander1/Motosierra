import torch
import soundfile as sf

from augmentation.audio_augmentation import AudioAugmentation


# Audio de prueba 
audio_path = "data/original/116765__unclesigmund__chainsaw-01.wav"


# Cargar audio
audio, sr = sf.read(audio_path)

print("Sample rate:")
print(sr)


# Convertir a tensor
audio = torch.tensor(audio, dtype=torch.float32)


audio = audio.unsqueeze(0)


print("\nAudio original:")
print(audio.shape)


# Crear aumentador
augment = AudioAugmentation()


# Aplicar Data Augmentation
audio_aug = augment(audio, sr)


print("\nAudio aumentado:")
print(audio_aug.shape)
