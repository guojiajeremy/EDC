# EDC

Official PyTorch Implementation of
"Encoder-Decoder Contrast for Unsupervised Anomaly Detection in Medical Images".

IEEE Transactions on Medical Imaging 2023. [paper](https://ieeexplore.ieee.org/document/10296925)

## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n my_env python=3.8.12
conda activate my_env
pip install -r requirements.txt
```
Experiments are conducted on NVIDIA GeForce RTX 3090 (24GB). Same GPU and package version are recommended. 

## 2. Prepare Datasets
Noted that `../` is the upper directory of this folder (EDC). It is where we keep all the datasets by default.

### OCT2017
Creat a new directory `../OCT2017`. Download ZhangLabData form [URL](https://data.mendeley.com/datasets/rscbjbr9sj/3).
Unzip the file, and move everything in `ZhangLabData/CellData/OCT` to `../OCT2017/`. The directory should be like:
```
|-- OCT2017
    |-- test
        |-- CNV
        |-- DME
        |-- DRUSEN
        |-- NORMAL
    |-- train
        |-- CNV
        |-- DME
        |-- DRUSEN
        |-- NORMAL
```

### APTOS
Creat a new directory `../APTOS`.
Download APTOS 2019 form [URL](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data).
Unzip the file to `../APTOS/original/`. Now, the directory would be like:
```
|-- APTOS
    |-- original
        |-- test_images
        |-- train_images
        |-- test.csv
        |-- train.csv
```
Run the following command to preprocess the data to `../APTOS/`.
```
python ./prepare_dataset/prepare_aptos.py --data-folder ../APTOS/original --save-folder ../APTOS
```
The directory would be like:
```
|-- APTOS
    |-- test
        |-- NORMAL
        |-- ABNORMAL
    |-- train
        |-- NORMAL
    |-- original
```
You can delete `original` if you want.

### ISIC2018
Creat a new directory `../ISIC2018`.
Go to the ISIC 2018 official [website](https://challenge.isic-archive.com/data/#2018).
Download "Training Data","Training Ground Truth", "Validation Data", and "Validation Ground Truth" of Task 3.
Unzip them to `../ISIC2018/original/`. Now, the directory would be like:
```
|-- ISIC2018
    |-- original
        |-- ISIC2018_Task3_Training_GroundTruth
        |-- ISIC2018_Task3_Training_Input
        |-- ISIC2018_Task3_Validation_GroundTruth
        |-- ISIC2018_Task3_Validation_Input
```
Run the following command to preprocess the data to `../ISIC2018/`.
```
python ./prepare_dataset/prepare_isic2018.py --data-folder ../ISIC2018/original --save-folder ../ISIC2018
```
The directory would be like:
```
|-- ISIC2018
    |-- test
        |-- NORMAL
        |-- ABNORMAL
    |-- train
        |-- NORMAL
    |-- original
```
You can delete `original` if you want.


### Br35H
Creat a new directory `../Br35H`.
Go to the ISIC 2018 official [website](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection).
Download "yes" and "no".
Unzip them to `../Br35H/original/`. Now, the directory would be like:
```
|-- Br35H
    |-- original
        |-- yes
        |-- no
```
Run the following command to preprocess the data to `../ISIC2018/`.
```
python ./prepare_dataset/prepare_br35h.py --data-folder ../Br35H/original --save-folder ../Br35H
```
The directory would be like:
```
|-- Br35H
    |-- test
        |-- NORMAL
        |-- ABNORMAL
    |-- train
        |-- NORMAL
    |-- original
```
You can delete `original` if you want.

## 3. Run Experiments
Run experiments with default arguments.

APTOS
```
python edc_aptos.py
```

OCT2017
```
python edc_oct.py
```

Br35H
```
python edc_br35h.py
```

ISIC2018
```
python edc_isic.py
```

### Further Improvement
See our new paper "ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction" NeurIPS 2023.
It introduces three key elements of contrastive learning into feature reconstruction, i.e., two-view contrastive pair,
global similarity, and stop gradient, building a fully 2-D contrastive paradigm. ReContrast also yields SOTA
performances on industrial UAD datasets (MVTecAD and VisA).


