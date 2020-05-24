
# Generator Function
sampling_rate = 22050
def generator(batch_size,is_true,cough_dir,non_cough_dir):
  if is_true:
    audio_list = train_files
  else:
    audio_list = val_files
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
      selected_audio,sampling_rate = read_audio(audio_directory,rand_audio)
      
      length = int(5.0 * sampling_rate)
      if len(selected_audio) > 110250:
        selected_audio = selected_audio[:length]

      elif len(selected_audio) < 110250:
        audio_n = np.zeros(length)
        rand_start = random.randrange(length - len(selected_audio))
        audio_n[rand_start:rand_start + len(selected_audio)] = selected_audio
        noise = np.random.randn(len(audio_n)) * 0.0005
        selected_audio = audio_n + noise

      volume_aug_audio = volume_aug(selected_audio)
      rand_noise = random.randrange(len(bg_noise_list))
      bg_noise = bg_noise_list[rand_noise]
      bg_noise_added = bg_noise_aug(volume_aug_audio,bg_noise)
      spectrum = melspectrogram(bg_noise_added,sampling_rate)
      batch_features[n] = normalize_spectrum(spectrum)
      batch_labels[n] = label

    yield batch_features,batch_labels