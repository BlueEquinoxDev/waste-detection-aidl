# Waste Segmentation and detection AIDL project

This repository contains the code developed by Martí Fabregat, Rafel Febrer, Ferran Miró-Gea and Miguel Ortiz in the scope of AIDL postgraduate course in UPC (Universitat Politècnica de Catalunya). With supervision of Amanda Duarte.

A vision transformer (ViT) is trained and evaluated to segment and classify waste. The model has been trained using [TACO Dataset](http://tacodataset.org) by Pedro F Proença and Pedro Simões. For more details check the paper: https://arxiv.org/abs/2003.06975

## Getting started

### Requirements 

Create a conda environment by running
```
conda create --name waste-management python=3.12.8
```

Then, activate the environment
```
conda activate waste-management
```

To install the required python packages simply type
```
pip3 install -r requirements_TACO.txt
```
### Download the dataset

To download the dataset images simply use:
```
python download.py
```

### Exploratory data analysis

Explore the notebook ``demo.ipynb``, modified version of the original notebook from the [TACO Repository](https://github.com/pedropro/TACO) that inspects the dataset.
The dataset is in COCO format. It contains the source pictures, anotations and labels. For more details related with the datasource please refer to [TACO Repository](https://github.com/pedropro/TACO).

### Split annotations in train, validation and test

To split the annotations for training and evaluation use ``split_dataset.py``. It has several optional flags.
```
python split_dataset.py --dataset_dir ./data [--test_percentage 0.1] [--val_percentage 0.1] [--seed 123] [--verbose False]
```
* Indicate the annotations directory using ``--dataset_dir``.
###### Optional:
* Use ``--test_percentage`` if you want to use a test split different than default 0.1 (10%).
* Use ``--val_percentage`` if you want to use a validation split different than default 0.1 (10%).
* Use ``--seed`` if you want to have a different random output. Default 123.
* Use ``--verbose`` if you want to have printed text on the console during execution.
* Use ``--taco28``  if you want generate a annotation.json file with 28 supercaegories
* Use ``--annotation_filename``  if you want to inform a diferent file of annotations to split

### Dataset Classes

#### Instance Segmentation - Mask R-CNN
The Taco Dataset for mask R-CNN class in ``custom_datasets/taco_dataset_mask_r_cnn_update.py`` has the functionality to load the Taco Dataset in for Instance Segmentation.

#### Image Classification - ViT for Viola
The Viola77 dataset for Image classification in ``custom_datasets/viola77_dataset.py`` has the functionality to load the Viola77 Dataset in for Image Classification.

#### Image Classification - ViT for Taco + Viola
The Viola77 dataset for Image classification in ``custom_datasets/viola77_dataset.py`` has the functionality to load the Viola77 Dataset in for Image Classification.

#### Hint
If when running ``datasets/taco_dataset.py`` you get the following error ``ModuleNotFoundError: No module named 'utilities'`` this can be solved by adding the project directory to the PYTHONPATH with ``export PYTHONPATH="/path/to/project:$PYTHONPATH"``

### Training

#### Training ViT in Viola Dataset
Run ``python run_classification_vit.py``

#### Training ViT in Viola + Taco Dataset
Run ``python run_classification_vit_viola_taco.py``

#### Training Mask R-CNN in Taco Dataset
Run ``python run_mask_r_cnn_update.py``


## Docker

Build the image with:
```docker build -t waste-detection-app .```

Run specific Python file:
```docker run --rm waste-detection-app <FILE_NAME.py>```



