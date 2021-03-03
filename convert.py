import librosa

#Importing Pytorch
import torch

#Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pydub import AudioSegment
import os

for dirname, dirnames, filenames in os.walk('mp3_files'):
    # print path to all filenames.
    for filename in filenames:
        path = os.path.join(dirname, filename).split('/')[-1]
        sound = AudioSegment.from_mp3("mp3_files/"+path)
        path = path.split('.')[0]+".wav"
        print(path)
        sound.export("wav_files/"+path, format="wav")