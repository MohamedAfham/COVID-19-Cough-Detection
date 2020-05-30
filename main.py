from src.DataGenerator.generator import train_generator
from src.DataGenerator.prepare_file import prepare
from src.Audio_Process.preprocess import PreProcess
from src.DataGenerator.val_test_arrays import val_test_data_array
from src.NeuralNetwork.cnn_model import neural_network
from src.NeuralNetwork.callback_func import adam_checkpoint,SGD_checkpoint, lr_schedule,stop_early
from tensorflow.compat.v1 import keras
import math

cough_dir = r"Sounds_up\Cough\\"
non_cough_dir = r"C:\Users\M A M Afham\Desktop\FSDKaggle2018\Non_Cough\\"
bg_dir = r"Sounds_up\Background Noise\\"

bg_noise_list = PreProcess().bg_noise_arr(bg_dir)

files_dict = prepare(cough_dir,non_cough_dir)
train_files = files_dict["train_files"]
val_files = files_dict["val_files"]
test_files = files_dict["test_files"]

directories = [non_cough_dir,cough_dir]
val_spectrum,val_label = val_test_data_array(val_files,True,directories,5,bg_noise_list)
test_spectrum,test_label = val_test_data_array(test_files,False,directories,5,bg_noise_list)


#Callbacks
model_path_Adam = r'Pre-trained Models\model-{val_accuracy:.2f}-Adam.h5'
checkpoint_Adam = adam_checkpoint(model_path_Adam)

model_path_SGD= r'Pre-trained Models\model-{val_accuracy:.2f}-SGD.h5'
checkpoint_SGD = SGD_checkpoint(model_path_SGD)

early_stop = stop_early(minimun_delta = 0.001, epoch_limit= 20)

def step_decay(epoch):
   initial_rate = 0.01
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_rate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
   return lrate

step_scheduler  = lr_schedule(step_decay)

#Model 
model = neural_network((128,287,1))

adam = keras.optimizers.Adam(learning_rate =0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer= adam, loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 15
epoch_step = int((len(train_files[0])+len(train_files[1]))/batch_size)
history = model.fit(train_generator(batch_size,train_files,5,cough_dir,non_cough_dir,bg_dir), epochs=50,steps_per_epoch=epoch_step,
                    validation_data = (val_spectrum,val_label), shuffle = True,callbacks = [checkpoint_Adam,early_stop,step_scheduler])