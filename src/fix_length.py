import numpy as np
import random

def make_audio_length(selected_audio,seconds,sampling_rate = 22050):
    length = int(seconds * sampling_rate)
    if len(selected_audio) > length:
        rand_start = random.randrange(len(selected_audio) - length)
        res_audio = selected_audio[rand_start:rand_start + length]

    elif len(selected_audio) < length:
        audio_n = np.zeros(length)
        rand_start = random.randrange(length - len(selected_audio))
        audio_n[rand_start:rand_start + len(selected_audio)] = selected_audio
        res_audio = audio_n

    else:
        res_audio = selected_audio
    noise = np.random.randn(len(res_audio)) * 0.0005
    res_audio = res_audio + noise

    return res_audio