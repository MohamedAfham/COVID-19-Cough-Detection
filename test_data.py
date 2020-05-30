import numpy as np 
import pandas as pd
from src.Audio_Process.fix_length import make_audio_length
from src.Audio_Process.preprocess import PreProcess
import os
from tensorflow.compat.v1.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

test_data = pd.read_csv(r"C:\Users\M A M Afham\Desktop\FSDKaggle2018\FSDKaggle2018.meta\test_post_competition_scoring_clips.csv")

test_coughs = test_data[test_data['label']=="Cough"]['fname'].tolist()
test_non_coughs = test_data[test_data['label']!="Cough"]['fname'].tolist()

print (len(test_coughs))
print (len(test_non_coughs))

model = load_model(r"Pre-trained Models\model-1.00-Adam.h5")

total_length = len(test_coughs) + len(test_non_coughs)

test_spectrum_list = []
test_label_list = []
audio_directory = r"C:\Users\M A M Afham\Desktop\FSDKaggle2018\FSDKaggle2018.audio_test\\"

for i in range(total_length):
    print (i,)
    if i < len(test_non_coughs):
        label = 0
        audio_file = test_non_coughs[i]
    else:
        label = 1
        audio_file = test_coughs[i - len(test_non_coughs)]

    selected_audio,fs = PreProcess().read_audio(audio_directory,audio_file)

    res_audio = make_audio_length(selected_audio,5,sampling_rate = 22050)

    spectrum = PreProcess().melspectrogram(res_audio,fs)
    spectrum = PreProcess().normalize_spectrum(spectrum)

    test_spectrum_list.append(spectrum)
    test_label_list.append(label)

test_features = np.array(test_spectrum_list)
test_labels = np.array(test_label_list)

pred = model.predict(x = test_features,batch_size = total_length)

print (model.evaluate(x = test_features, y = test_labels,batch_size=total_length))

preds = np.array([int(np.round(pred[i])) for i in range(len(pred))])


print (confusion_matrix(test_labels,preds))
print ('\n')
print (classification_report(test_labels,preds))