import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf


class MotosierraDataset(Dataset):

    def __init__(self, csv_file, audio_dir, transform=None):

        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform

        # Crear etiquetas automáticamente
        categories = sorted(self.data["category"].unique())

        self.label_map = {
            category: idx
            for idx, category in enumerate(categories)
        }

        print("Clases encontradas:")
        print(self.label_map)

        
        self.target_length = 44100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        # Ruta completa del audio
        audio_path = os.path.join(
            self.audio_dir,
            row["filename"]
        )

        # Leer audio
        audio, sr = sf.read(audio_path)

        # Convertir a tensor
        audio = torch.tensor(
            audio,
            dtype=torch.float32
        )

        
        if audio.ndim > 1:
            audio = audio.mean(dim=1)

        
        # Normalizar longitud
        

        if audio.shape[0] > self.target_length:
            # Recortar
            audio = audio[:self.target_length]

        elif audio.shape[0] < self.target_length:
            # Rellenar con ceros
            padding = self.target_length - audio.shape[0]
            audio = F.pad(audio, (0, padding))

    
        audio = audio.unsqueeze(0)

        # Aplicar transformación 
        if self.transform is not None:
            audio = self.transform(audio)

        # Obtener etiqueta
        label = self.label_map[row["category"]]

        return audio, label
