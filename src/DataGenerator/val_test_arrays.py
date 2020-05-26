def val_test_data_array(files,is_val,directories,seconds,bg_noise_list):
    from src.Audio_Process.fix_length import make_audio_length
    from src.Audio_Process.preprocess import PreProcess

    """
    files: The list of audio file names in the directory
    is_val: validation dataset or test_dataset need (Bool)
    directories: [non_cough audio directory, cough audio directory]
    seconds: Length of an audio in seconds
    bg_noise_list: Array/List of background Noise

    Returns:
    list of a spectrums and list of corresponding labels
    """

    total_length = len(files[0])+len(files[1])
    spectrum_list = []
    label_list = []
    for i in range(total_length):
        if i < len(files[0]):
            label = 0
        else:
            label = 1
        audio_directory = directories[label]  # 0 for non_cough and 1 for cough
        selected_audio,fs = PreProcess().read_audio(audio_directory,files[label][i - len(files[0]) * label])

        res_audio = make_audio_length(selected_audio,seconds,sampling_rate = 22050)
        
        if is_val:
            vol_aug = PreProcess().volume_aug(res_audio)
            bg_noise_added = PreProcess().bg_noise_aug(vol_aug,bg_noise_list)
            spectrum = PreProcess().melspectrogram(bg_noise_added,fs)
            spectrum = PreProcess().normalize_spectrum(spectrum)
        else:
            spectrum = PreProcess().melspectrogram(res_audio,fs)
            spectrum = PreProcess().normalize_spectrum(spectrum)

        spectrum_list.append(spectrum)
        label_list.append(label)
    return (spectrum_list,label_list)

        








        