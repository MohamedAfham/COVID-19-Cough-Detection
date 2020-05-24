import random
import os

def prepare(cough_dir,non_cough_dir):
    cough = os.listdir(cough_dir)
    non_cough =os.listdir(non_cough_dir)

    test_cough = random.sample(cough,45)
    test_non_cough = random.sample(non_cough,45)

    test_files = [test_non_cough, test_cough]

    train_val_cough = list(set(cough) - set(test_cough))
    train_val_non_cough = list(set(non_cough) - set(test_non_cough))

    random.shuffle(train_val_cough)
    train_cough,val_cough = train_val_cough[:int(len(train_val_cough)*0.9)], train_val_cough[int(len(train_val_cough)*0.9):]

    random.shuffle(train_val_non_cough)
    train_non_cough,val_non_cough = train_val_non_cough[:int(len(train_val_non_cough)*0.9)], train_val_non_cough[int(len(train_val_non_cough)*0.9):]

    train_files = [train_non_cough, train_cough]
    val_files = [val_non_cough, val_cough]

    files_dict = {"train_files":train_files, "val_files":val_files, "test_files":test_files}
    
    return files_dict