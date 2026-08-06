from torch.utils.data import DataLoader

from dataset import MotosierraDataset


# Crear Dataset
dataset = MotosierraDataset(
    csv_file="data/metadata.csv",
    audio_dir="data/original",
    train=True
)


# Crear DataLoader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)


print("Cantidad de batches:")

print(len(loader))


for mel, labels in loader:

    print("\nBatch de Mel:")

    print(mel.shape)

    print("\nBatch Labels:")

    print(labels)

    break