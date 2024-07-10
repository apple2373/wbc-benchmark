# White Blood Cell Classification under Domain Shift

If you find the code useful, please consider citing the following paper: 
- https://arxiv.org/abs/2303.01777
- Satoshi Tsutsui, Zhengyang Su, and Bihan Wen. (2023). Benchmarking White Blood Cell Classification Under Domain Shift. IEEE International Conference on Acoustics, Speech, & Signal Processing (ICASSP).
```
@inproceedings{tsutsui2023wbc,
	author = {Satoshi Tsutsui and Zhengyang Su and Bihan Wen},
	booktitle = {IEEE International Conference on Acoustics, Speech, & Signal Processing (ICASSP)},
	month = {06},
	title = {Benchmarking White Blood Cell Classification Under Domain Shift},
	year = {2023}
}
```

## Dataset Preparation
- RaabinWBC
	- Download and decompress `Train`, `TestA`, and `TestB` into `./data/RaabinWBC/` from https://raabindata.com/free-data/.
- LISC
	- Download and decompress `LISC` into `./data/` from https://drive.google.com/file/d/1gknVrSs1CRy8PoIh1HXiGu-1ObH3cQ9S, which is provided by https://github.com/nimaadmed/WBC_Feature. 
	- then `python3 scripts/organize_lisc.py`, which will make `./data/LISCCropped`
- BCCD
	- Download and decompress `BCCD` into `./data/`  from https://github.com/apple2373/BCCD 
	- `cd data; git clone https://github.com/apple2373/BCCD; cd ../`

## Environment
- We used python version 3.8.
- We used CUDA version 11.6
- see [`requirements.txt`](./requirements.txt).
    - We use pytorch with version 1.12.1. 
    - We use torchvision with version 0.13.1.
    - We use timm with version 0.6.11. 
- We used miniconda. The following commands should do the job.
	```
	conda create -y --name=wbc-benchmark python=3.8
	conda activate wbc-benchmark
	pip install numpy==1.23.1 pandas==1.5.0 Pillow==9.3.0 tqdm==4.64.1
	conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
	# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 # if conda is too slow...
	pip install timm==0.6.11
	```
- The exact enviroment we use is see [`environment.yml`](./environment.yml).
    

## Usage
- See `python3 main.py --help` for details.
- By default, it will train on RaabinWBC dataset, and test on LISC dataset.
- E.g., to train ResNet50 with Group Normalization
    - `python3 main.py --backbone resnet50_gn`
- E.g., to test on RabbinWBC Test A set.
    - `python3 main.py --backbone resnet50_gn --test-img-root ./data/RaabinWBC/TestA`
- Trained models can be tested on other datasets by feeding `--epochs 0 --resume <path to pretrained model> --test-img-root <path to test set root>`.

## Models
All models are initialized with ImageNet pretrained weights.

Models used in the paper.
- vgg16
	- VGG16
	- `python3 main.py --backbone vgg16` 
- vgg16_bn
	- VGG16 with Batch Normalization
	- `python3 main.py --backbone vgg16_bn` 
- resnet50
	- ResNet50
	- `python3 main.py --backbone resnet50` 
- resnet50-gn
	- ResNet50 that replaced Batch Normalization with Group Normalization
	- `python3 main.py --backbone resnet50_gn` 
- resnet50-fc
	- ResNet50 + VGG-style fully-connected layers.
	- `python3 main.py --backbone resnet50 --addfc 1` 
- resnet50-freezebn
	- ResNet50 + freeze batch normalization layers with ImageNet pretrained parameters
	- `python3 main.py --backbone resnet50 --freezebn 1` 
- resnet50-freezebn-last16only
	- ResNet50 + freeze batch normalization layers with ImageNet pretrained parameters + fine-tune only the last 16 layers.
	- `python3 main.py --backbone resnet50 --freezebn 1 --last16only 1` 

Models not used in the paper.
- resnet50-gn-mixup
	- ResNet50 that replaced Batch Normalization with Group Normalization
	- `python3 main.py --backbone resnet50_gn --mixup 1` 
- vit_b_16
	- Vision Transformer (ViT) Base 16x16.
	- `python3 main.py --backbone vit_b_16` 

## Accuracy (%)
The results are obtained by running the model 10 times with random seeds of 0-9. See [`train_resnet50_gn.sh`](scripts/train_resnet50_gn.sh) for example. We report mean accuracy with 95% confidence interval.
- vgg16
	- testA: 98.75 +- 0.06
	- testB: 64.79 +- 6.05
	- LISC: 74.44 +- 2.72
	- BCCD: 17.32 +- 3.40
- vgg16-bn
	- testA: 98.64 +- 0.18
	- testB: 12.06 +- 5.46
	- LISC: 32.33 +- 6.17
	- BCCD: 19.86 +- 2.23
- resnet50
	- testA: 98.53 +- 0.18
	- testB: 34.55 +- 10.72
	- LISC: 23.11 +- 3.04
	- BCCD: 23.19 +- 7.87
- resnet50-fc
	- testA: 98.46 +- 0.19
	- testB: 35.63 +- 4.51
	- LISC: 27.74 +- 3.44
	- BCCD: 21.34 +- 3.04
- resnet50-freezebn
	- testA: 98.94 +- 0.06
	- testB: 65.38 +- 2.78
	- LISC: 51.48 +- 7.14
	- BCCD: 11.97 +- 2.46
- resnet50-freezebn-last16only
	- testA: 98.67 +- 0.12
	- testB: 61.92 +- 2.11
	- LISC: 74.24 +- 2.46
	- BCCD: 23.33 +- 1.21
- resnet50-gn
	- testA: 98.24 +- 0.14
	- testB: 77.95 +- 5.43
	- LISC: 73.07 +- 2.07
	- BCCD: 49.12 +- 6.94
- resnet50-gn-mixup
	- testA: 98.44 +- 0.09
	- testB: 19.67 +- 4.71
	- LISC: 63.50 +- 2.65
	- BCCD: 53.11 +- 4.82
- vit_b_16
	- testA: 98.33 +- 0.14
	- testB: 85.06 +- 2.74
	- LISC: 69.77 +- 3.09
	- BCCD: 41.00 +- 6.66
