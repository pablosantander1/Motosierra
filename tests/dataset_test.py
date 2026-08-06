import torch

from dataset import MotosierraDataset


dataset = MotosierraDataset(
    csv_file="data/metadata.csv",
    audio_dir="data/original",
    train=True
)


print("Cantidad de audios:")
print(len(dataset))


mel, label = dataset[0]


print("\nMel:")
print(mel.shape)


print("\nLabel:")
print(label)