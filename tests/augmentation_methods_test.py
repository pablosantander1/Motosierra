import torch
import soundfile as sf

from augmentation.audio_augmentation import AudioAugmentation


# Audio de prueba
audio_path = "data/original/116765__unclesigmund__chainsaw-01.wav"


# Cargar audio
audio, sr = sf.read(audio_path)


# Convertir a tensor
audio = torch.tensor(
    audio,
    dtype=torch.float32
)


# Agregar canal
audio = audio.unsqueeze(0)


print("Audio original:")
print(audio.shape)

print("Sample rate:")
print(sr)


# Crear aumentador
augment = AudioAugmentation()


# ==========================
# 1) Ruido
# ==========================

print("\n--- Ruido ---")

audio_noise = augment.add_noise(audio)

print(audio_noise.shape)


# ==========================
# 2) Time Shift
# ==========================

print("\n--- Time Shift ---")

audio_shift = augment.time_shift(audio)

print(audio_shift.shape)


# ==========================
# 3) Cambio de volumen
# ==========================

print("\n--- Cambio de volumen ---")

audio_volume = augment.change_volume(audio)

print(audio_volume.shape)


# ==========================
# 4) Pitch Shift
# ==========================

print("\n--- Pitch Shift ---")

audio_pitch = augment.pitch_shift(
    audio,
    sr
)

print(audio_pitch.shape)


# ==========================
# 5) Time Stretch
# ==========================

print("\n--- Time Stretch ---")

audio_speed = augment.time_stretch(
    audio,
    sr
)

print(audio_speed.shape)