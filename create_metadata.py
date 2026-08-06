import os
import pandas as pd


audio_dir = "data/original"

rows = []


for filename in os.listdir(audio_dir):

    if filename.endswith(".wav"):

        rows.append({
            "filename": filename,
            "category": "chainsaw"
        })


df = pd.DataFrame(rows)


df.to_csv(
    "data/metadata.csv",
    index=False
)


print("CSV creado")
print(df.head())
print("Cantidad de audios:", len(df))