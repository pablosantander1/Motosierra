import os
import shutil
import random
import pandas as pd

# CONFIGURACIÓN

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CHAINSAW_SOURCE = os.path.join(PROJECT_DIR, "data", "original")

ESC50_AUDIO = r"C:\Users\maria\OneDrive\Escritorio\Doc Pablo\ESC-50-master\ESC-50-master\audio"

ESC50_META = r"C:\Users\maria\OneDrive\Escritorio\Doc Pablo\ESC-50-master\ESC-50-master\meta\esc50.csv"

CHAINSAW_DEST = os.path.join(PROJECT_DIR, "data", "chainsaw")

NO_CHAINSAW_DEST = os.path.join(PROJECT_DIR, "data", "no_chainsaw")

OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "metadata.csv")

NEGATIVE_SAMPLES = 450

RANDOM_SEED = 42


# CREAR CARPETAS

os.makedirs(CHAINSAW_DEST, exist_ok=True)
os.makedirs(NO_CHAINSAW_DEST, exist_ok=True)

metadata = []

# COPIAR AUDIOS DE MOTOSIERRA

chainsaw_files = [
    f for f in os.listdir(CHAINSAW_SOURCE)
    if f.lower().endswith(".wav")
]

for file in chainsaw_files:

    shutil.copy2(
        os.path.join(CHAINSAW_SOURCE, file),
        os.path.join(CHAINSAW_DEST, file)
    )

    metadata.append({
        "filename": f"chainsaw/{file}",
        "category": "chainsaw"
    })

print(f"Motosierras: {len(chainsaw_files)}")


# ESC-50


esc50 = pd.read_csv(ESC50_META)

negative = esc50[
    esc50["category"] != "chainsaw"
].copy()

random.seed(RANDOM_SEED)

negative = negative.sample(
    n=min(NEGATIVE_SAMPLES, len(negative)),
    random_state=RANDOM_SEED
)


for _, row in negative.iterrows():

    file = row["filename"]

    shutil.copy2(
        os.path.join(ESC50_AUDIO, file),
        os.path.join(NO_CHAINSAW_DEST, file)
    )

    metadata.append({
        "filename": f"no_chainsaw/{file}",
        "category": "no_chainsaw"
    })

print(f"No motosierras: {len(negative)}")

# GUARDAR 

metadata = pd.DataFrame(metadata)

metadata.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nDataset generado correctamente.\n")

print(metadata["category"].value_counts())

print("\nArchivo guardado en:")

print(OUTPUT_CSV)
