# Reading audio clip
import librosa

#Importing Pytorch
import torch

import numpy as np

#Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pydub import AudioSegment
import os

import IPython.display as display
#display.Audio("wav_files/afrikaans1.wav", autoplay=True)

for dirname, dirnames, filenames in os.walk('wav_files'):
    # print path to all filenames.
    for filename in filenames:
        print("----------")
        print("----------"+filename+"----------")
        print("----------")
        filename = "wav_files/" + filename
        audio, rate = librosa.load(filename, sr = 16000)

        # printing
        #print("audio", audio)
        #print("rate", rate)


        # Importing Wav2Vec pretrained model
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # Taking an input value
        input_values = tokenizer(audio, return_tensors = "pt").input_values

        #print("----input values-------")
        #print(input_values)
        inp = np.array(input_values[0])
        #print(inp)
        #print("shape", inp.shape)


        # Storing logits (non-normalized prediction values)
        logits = model(input_values).logits

        #Passing the logit values to softmax to get the predicted values
        # Storing predicted ids
        prediction = torch.argmax(logits, dim = -1)

        pred = np.array(prediction)
        prde = np.array(prediction[0])


        print("----predicted-------")
        print(prde)
        print("shape", prde.shape)
        # Passing the prediction to the tokenzer decode to get the transcription
        transcription = tokenizer.batch_decode(prediction)[0]

        # Printing the transcription
        print(transcription)