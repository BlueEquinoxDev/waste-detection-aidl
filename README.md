# Waste Segmentation and detection AIDL project

This repository contains the code developed by Martí Fabregat, Rafel Febrer, Ferran Miró-Gea and Miguel Ortiz in the scope of AIDL postgraduate course in UPC (Universitat Politècnica de Catalunya). With supervision of Amanda Duarte.

Several models for instance segmentation and image classification have been trained and evaluated to segment and classify waste.

The model has been trained using [TACO Dataset](http://tacodataset.org) by Pedro F Proença and Pedro Simões. For more details check the paper: https://arxiv.org/abs/2003.06975

The [Viola77](https://huggingface.co/datasets/viola77data/recycling-dataset) dataset is used as well for classification. Under Apache 2.0 License.

## Models used are:
### Classification:
- Restnet
- ViT: Finetunning based on [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
### Instance segmentation
- Mask R-CNN
- Mask2Former


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
python split_dataset.py --dataset_dir ./data [--test_percentage 0.1] [--val_percentage 0.1] [--seed 123] [--verbose False] [--]
```
* Indicate the annotations directory using ``--dataset_dir``.
* Indicate the dataset to use ``--dataset_type``. It depends on the task to do. It can be:
    * ``classification`` for classification tasks (Use it to run classification in taco28, taco5, taco30viola11).
    * ``taco28`` for segmentation in taco28 dataset (Taco dataset with 28 categories, includes all data)
    * ``taco5`` for segmentation in taco5 dataset (Taco dataset with a subsample of 5 categories)
###### Optional:
* Use ``--test_percentage`` if you want to use a test split different than default 0.1 (10%).
* Use ``--val_percentage`` if you want to use a validation split different than default 0.1 (10%).
* Use ``--seed`` if you want to have a different random output. Default 123.
* Use ``--verbose`` if you want to have printed text on the console during execution.

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



