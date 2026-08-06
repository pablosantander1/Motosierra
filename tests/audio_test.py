import os
import soundfile as sf

DATASET = "data/original"

# Obtener todos los archivos WAV
files = [f for f in os.listdir(DATASET) if f.lower().endswith(".wav")]

print(f"Cantidad de audios encontrados: {len(files)}")

if len(files) == 0:
    print("No se encontraron archivos WAV.")
    exit()

# Leer el primer audio
audio_path = os.path.join(DATASET, files[0])

audio, sample_rate = sf.read(audio_path)

print("\nPrimer archivo:")
print(files[0])

print("\nFrecuencia de muestreo:")
print(sample_rate)

print("\nForma del audio:")
print(audio.shape)

print("\nDuración:")
print(round(len(audio) / sample_rate, 3), "segundos")