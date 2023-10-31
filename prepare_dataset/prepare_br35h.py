import os
import random
from shutil import copyfile
import pandas as pd
import numpy as np
import cv2
import argparse

random.seed(1)

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data-folder', default='/data/disk2T1/guoj/Br35H/original', type=str)
parser.add_argument('--save-folder', default='/data/disk2T1/guoj/Br35H', type=str)
config = parser.parse_args()

source_dir = config.data_folder
target_dir = config.save_folder

train_normal_path = []
valid_normal_path = []
valid_abnormal_path = []

files = os.listdir(os.path.join(source_dir, 'no'))
for file in files:
    if int("".join(filter(str.isdigit, file))) < 1000:
        train_normal_path.append(os.path.join(source_dir, 'no', file))
    else:
        valid_normal_path.append(os.path.join(source_dir, 'no', file))

files = os.listdir(os.path.join(source_dir, 'yes'))
for file in files:
    valid_abnormal_path.append(os.path.join(source_dir, 'yes', file))

target_train_normal_dir = os.path.join(target_dir, 'train', 'NORMAL')
if not os.path.exists(target_train_normal_dir):
    os.makedirs(target_train_normal_dir)

target_test_normal_dir = os.path.join(target_dir, 'test', 'NORMAL')
if not os.path.exists(target_test_normal_dir):
    os.makedirs(target_test_normal_dir)

target_test_abnormal_dir = os.path.join(target_dir, 'test', 'ABNORMAL')
if not os.path.exists(target_test_abnormal_dir):
    os.makedirs(target_test_abnormal_dir)

for f in train_normal_path:
    copyfile(f, os.path.join(target_train_normal_dir, os.path.basename(f)))

for f in valid_normal_path:
    copyfile(f, os.path.join(target_test_normal_dir, os.path.basename(f)))

for f in valid_abnormal_path:
    copyfile(f, os.path.join(target_test_abnormal_dir, os.path.basename(f)))
