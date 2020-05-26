from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler,Callback
import math

def adam_checkpoint(path):
    trained_model_path_Adam = path + 'model-{val_accuracy:.2f}-Adam.h5'
    checkpoint_Adam = ModelCheckpoint(trained_model_path_Adam, 
                                monitor='val_accuracy', 
                                save_best_only=True, 
                                mode='max', 
                                period=1)
    return checkpoint_Adam

def SGD_checkpoint(path):
    trained_model_path_SGD = path + 'model-{val_accuracy:.2f}-SGD.h5'
    checkpoint_SGD = ModelCheckpoint(trained_model_path_SGD, 
                                monitor='val_accuracy', 
                                save_best_only=True, 
                                mode='max', 
                                period=1)
    return checkpoint_SGD

def stop_early():
    early_stop = EarlyStopping(monitor='val_accuracy', 
                            min_delta=0.001, 
                            patience=30, 
                            mode='max',
                            restore_best_weights=False)

    return early_stop

def lr_schedule(func):
    step_scheduler  = LearningRateScheduler(func)
    return step_scheduler