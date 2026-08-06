import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MotosierraDataset
from models.cnn_audio import CNNAudio
from augmentation.audio_transform import AudioTransform


# ==========================
# Configuración
# ==========================

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001


# ==========================
# Selección dispositivo
# ==========================

if torch.cuda.is_available():

    device = torch.device("cuda")

    print(
        "Entrenando con GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    device = torch.device("cpu")

    print("Entrenando con CPU")


# ==========================
# Dataset
# ==========================

dataset = MotosierraDataset(

    csv_file="data/metadata.csv",

    audio_dir="data"

)


train_loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)


print(
    "Cantidad de audios:",
    len(dataset)
)


# ==========================
# Transformación Mel
# ==========================
transform = AudioTransform()
transform = transform.to(device)


# ==========================
# Modelo
# ==========================

model = CNNAudio(

    num_classes=2

)


model = model.to(device)


# ==========================
# Pérdida
# ==========================

criterion = nn.CrossEntropyLoss()


# ==========================
# Optimizador
# ==========================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)



print("\nInicio del entrenamiento...\n")


# ==========================
# Entrenamiento
# ==========================

for epoch in range(EPOCHS):


    model.train()


    running_loss = 0

    correct = 0

    total = 0



    for audio, label in train_loader:


        audio = audio.to(device)

        label = label.to(device)



        # Audio -> Mel Spectrogram

        mel = transform(audio)



        optimizer.zero_grad()



        output = model(mel)



        loss = criterion(
            output,
            label
        )



        loss.backward()



        optimizer.step()



        running_loss += loss.item()



        _, predicted = torch.max(
            output,
            1
        )


        total += label.size(0)


        correct += (
            predicted == label
        ).sum().item()



    accuracy = (
        100 * correct / total
    )



    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )



# ==========================
# Guardar modelo
# ==========================

torch.save(

    model.state_dict(),

    "models/cnn_motosierra.pth"

)


print(
    "\nModelo guardado correctamente"
)