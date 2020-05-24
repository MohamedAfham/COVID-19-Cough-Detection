import librosa
import random
import numpy as np
import os

class PreProcess():    
    def read_audio(self,audio_directory,audio_file):
        audio,sampling_rate = librosa.core.load(audio_directory+audio_file)
        return (audio,sampling_rate)

    def volume_aug(self,audio):
        factor_list = [0.5,1,1.5,2]
        factor = random.choice(factor_list)
        augmented_samples = audio * factor
        return augmented_samples

    def bg_noise_aug(self,audio):
        bg_dir = "Sounds_up\\Background Noise\\"
        bg_noise_list = os.listdir(bg_dir)
        bg_noise = self.read_audio(bg_dir,random.choice(bg_noise_list))[0]
        length = len(audio)
        rand_start = random.randrange(len(bg_noise)- length)
        augmented_audio = audio + bg_noise[rand_start:rand_start + length]
        return augmented_audio

    def melspectrogram(self,audio,sampling_rate):  #no of mels means no of mel filters
        #samples, sampling_rate = librosa.core.load(directory + audio_file)   #loading data set
        samples = audio
        frame_size = 0.025
        frame_stride = frame_size * 0.7
        no_of_mels = 128
        frame_length, frame_step = frame_size * sampling_rate, frame_stride * sampling_rate  
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        samples_length = len(samples)
        num_frames = int(np.ceil(float(np.abs(samples_length - frame_length)) / frame_step))

        pad_samples_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_samples_length - samples_length))
        pad_samples = np.append(samples, z)  #zero padding
            
        spect_samples = np.abs(librosa.stft(pad_samples, n_fft=frame_length, hop_length=frame_step, window=np.hamming(frame_length))) #calculating stft 
        spect_samples = librosa.amplitude_to_db(spect_samples, ref=np.max)
        mel_filter = librosa.filters.mel(sampling_rate, frame_length, n_mels=128, fmin=0, fmax=None)  #calculating mel filter array
        
        mel_spect_samples = np.dot(mel_filter, spect_samples)   #calculating mel spectrogram
        spectrum = mel_spect_samples.ravel().reshape(128,287,1)
        #spectrum = mel_spect_samples
        return spectrum

    def normalize_spectrum(self,spectrum):
        return (spectrum - np.mean(spectrum))/np.std(spectrum)


    
