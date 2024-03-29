from __future__ import print_function
import zipfile
import os
import PIL

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 32 x 32 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    # transforms.Normalize((0.5, ), ( 0.5,))
])

def spec_trans(specific_transform):
    trans = transforms.Compose([
        specific_transform,
        transforms.Resize((32, 32)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, ), ( 0.5,))
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])
    return trans

def spec_trans_end(specific_transform):
    trans = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, ), ( 0.5,))
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
        specific_transform
    ])
    return trans    

randoTrans = transforms.Compose([
    transforms.RandomApply(
        [
            transforms.ColorJitter(brightness=0.6),
            transforms.ColorJitter(contrast=0.7),
            transforms.ColorJitter(hue=0.5),
            transforms.RandomAffine(90),
            transforms.RandomAffine(0, translate=((0.40,0.40))),
            transforms.RandomAffine(0, shear=10),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0)
        ]
        ,p=0.3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
        transforms.RandomErasing(p=0.9, value='random')
])

def initialize_data(folder):
    # train_zip = folder + '/train_images.zip'
    # test_zip = folder + '/test_images.zip'
    # if not os.path.exists(train_zip) or not os.path.exists(test_zip):
    #     raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
    #           + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2018/data '))
    # # extract train_data.zip to train_data
    train_folder = folder + '/train_images'
    # if not os.path.isdir(train_folder):
    #     print(train_folder + ' not found, extracting ' + train_zip)
    #     zip_ref = zipfile.ZipFile(train_zip, 'r')
    #     zip_ref.extractall(folder)
    #     zip_ref.close()
    # # extract test_data.zip to test_data
    test_folder = folder + '/test_images'
    # if not os.path.isdir(test_folder):
    #     print(test_folder + ' not found, extracting ' + test_zip)
    #     zip_ref = zipfile.ZipFile(test_zip, 'r')
    #     zip_ref.extractall(folder)
    #     zip_ref.close()

    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            if dirs.startswith('000'):
                print('---',dirs)
                os.mkdir(val_folder + '/' + dirs)
                for f in os.listdir(train_folder + '/' + dirs):
                    print(f)
                    if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
                        # move file to validation folder
                        os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
