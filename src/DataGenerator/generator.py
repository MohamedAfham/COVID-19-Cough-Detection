# Generator Function
import numpy as np
import random

def train_generator(batch_size,train_files,seconds,cough_dir,non_cough_dir,bg_noise_dir):
  from src.Audio_Process.preprocess import PreProcess
  from src.Audio_Process.fix_length import make_audio_length

  audio_list = train_files
  bg_noise_list = PreProcess().bg_noise_arr(bg_noise_dir)

  while True:
    batch_features = np.zeros((batch_size,128,287,1))
    batch_labels = np.zeros((batch_size,1))

    for n in range(batch_size):
      label = np.random.randint(0,2)

      if label == 1:
        audio_directory = cough_dir
      else:
        audio_directory = non_cough_dir

      rand_audio = random.choice(audio_list[label])
      selected_audio,sampling_rate = PreProcess().read_audio(audio_directory,rand_audio)
      
      res_audio = make_audio_length(selected_audio,seconds)

      volume_aug_audio = PreProcess().volume_aug(res_audio)
      bg_noise_added = PreProcess().bg_noise_aug(volume_aug_audio,bg_noise_list)
      spectrum = PreProcess().melspectrogram(bg_noise_added,sampling_rate)

      batch_features[n] = PreProcess().normalize_spectrum(spectrum)
      batch_labels[n] = label

    yield batch_features,batch_labels