import torch

from models.cnn_audio import CNNAudio

model = CNNAudio(num_classes=1)

x = torch.randn(8, 1, 64, 87)

y = model(x)

print("Entrada:", x.shape)
print("Salida:", y.shape)