import torch
import soundfile as sf

from augmentation.audio_augmentation import AudioAugmentation
from augmentation.spec_augmentation import SpecAugmentation
from utils.audio_transform import AudioTransform

audio_path = "data/original/116765__unclesigmund__chainsaw-01.wav"

audio, sr = sf.read(audio_path)

audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

print("Audio:", audio.shape)

augment = AudioAugmentation()

audio = augment(audio, sr)

transform = AudioTransform()

mel = transform(audio)

print("Mel:", mel.shape)

spec_aug = SpecAugmentation()

mel = spec_aug(mel)

print("Mel aumentado:", mel.shape)