# Waste Segmentation and detection AIDL project

This repository contains the code developed by Martí Fabregat, Rafel Febrer, Ferran Miró-Gea and Miguel Ortiz in the scope of AIDL postgraduate course in UPC (Universitat Politècnica de Catalunya). With supervision of Amanda Duarte.

Several models for instance segmentation and image classification have been trained and evaluated to segment and classify waste.

## Table of Contents
- [Getting started](#getting-started)
  - [Requirements](#requirements)
  - [Datasets](#datasets)
    - [TACO Dataset](#taco-dataset)
    - [Viola77](#viola77)
  - [Exploratory data analysis](#exploratory-data-analysis)
- [Training](#training)
  - [Image Classification with ResNet-50](#image-classification-with-resnet-50)
    - [Split dataset](#split-dataset)
    - [Dataset classes](#dataset-classes)
      - [ResNet-50 for Viola77](#resnet-50-for-viola77)
      - [ResNet-50 for Taco](#resnet-50-for-taco)
    - [Train](#train)
    - [Evaluate](#evaluate)
    - [Results](#results)
  - [Image Classification with ViT](#image-classification-with-vit)
    - [Split dataset](#split-dataset-1)
    - [Dataset classes](#dataset-classes-1)
      - [ViT for Viola77](#vit-for-viola77)
      - [ViT for Taco + Viola77](#vit-for-taco--viola77)
    - [Train](#train-1)
    - [Evaluate](#evaluate-1)
    - [Results](#results-1)
  - [Instance segmentation with Mask R-CNN](#instance-segmentation-with-mask-r-cnn)
    - [Split dataset](#split-dataset-2)
    - [Dataset classes](#dataset-classes-2)
    - [Train](#train-2)
    - [Evaluate](#evaluate-2)
    - [Results](#results-2)
  - [Instance segmentation with Mask2Former](#instance-segmentation-with-mask2former)
    - [Split dataset](#split-dataset-3)
    - [Dataset classes](#dataset-classes-3)
    - [Train](#train-3)
    - [Evaluate](#evaluate-3)
    - [Results](#results-3)
- [MLOps](#mlops)
  - [Running the application with Docker](#running-the-application-with-docker)
  - [Google Cloud](#google-cloud)
  - [API](#api)
- [Demo](#demo)

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

### Datasets

#### TACO Dataset

The model has been trained using [TACO Dataset](http://tacodataset.org) by Pedro F Proença and Pedro Simões. For more details check the paper: https://arxiv.org/abs/2003.06975

To download the dataset images simply use:
```
python download.py
```

#### Viola77

The [Viola77](https://huggingface.co/datasets/viola77data/recycling-dataset) dataset is used as well for classification. Under Apache 2.0 License.


### Exploratory data analysis

Explore the notebook ``demo.ipynb``, modified version of the original notebook from the [TACO Repository](https://github.com/pedropro/TACO) that inspects the dataset.
The dataset is in COCO format. It contains the source pictures, anotations and labels. For more details related with the datasource please refer to [TACO Repository](https://github.com/pedropro/TACO).

## Training

### Image Classification with ResNet-50
#### Split dataset
#### Dataset classes
##### ResNet-50 for Viola77
##### ResNet-50 for Taco
#### Train
#### Evaluate
#### Results

### Image Classification with ViT
#### Split dataset
#### Dataset classes
##### ViT for Viola77
The Viola77 dataset for Image classification in ``custom_datasets/viola77_dataset.py`` has the functionality to load the Viola77 Dataset in for Image Classification.

##### ViT for Taco + Viola77
The Viola77 dataset for Image classification in ``custom_datasets/viola77_dataset.py`` has the functionality to load the Viola77 Dataset in for Image Classification.
#### Train
Run ``python run_classification_vit.py``
Run ``python run_classification_vit_viola_taco.py``
#### Evaluate
#### Results

### Instance segmentation with Mask R-CNN
#### Split dataset
To split the annotations for training and evaluation use ``split_dataset.py``. It has several optional flags.
```
python split_dataset.py --dataset_dir ./data [--test_percentage 0.1] [--val_percentage 0.1] [--seed 123] [--verbose False] [--]
```
* Indicate the annotations directory using ``--dataset_dir``.
* Indicate the dataset to use ``--dataset_type``. It depends on the task to do. It can be:
    * ``classification`` for classification tasks (Use it to run classification in taco28, taco5, taco30viola11).
    * ``taco28`` for segmentation in taco28 dataset (Taco dataset with 28 categories, includes all data)
    * ``taco5`` for segmentation in taco5 dataset (Taco dataset with a subsample of 5 categories)
##### Optional:
* Use ``--test_percentage`` if you want to use a test split different than default 0.1 (10%).
* Use ``--val_percentage`` if you want to use a validation split different than default 0.1 (10%).
* Use ``--seed`` if you want to have a different random output. Default 123.
* Use ``--verbose`` if you want to have printed text on the console during execution.
#### Dataset classes
The Taco Dataset for mask R-CNN class in ``custom_datasets/taco_dataset_mask_r_cnn_update.py`` has the functionality to load the Taco Dataset in for Instance Segmentation.
#### Train
Run ``python run_mask_r_cnn_update.py``
#### Evaluate
#### Results

### Instance segmentation with Mask2Former
#### Split dataset
#### Dataset classes
#### Train
#### Evaluate
#### Results


## MLOps

### Running the application with Docker
Build the image with:
```docker build -t waste-detection-app .```

Run specific Python file:
```docker run --rm waste-detection-app <FILE_NAME.py>```

### Google Cloud
This repository automates the setup of the GCP infrastructure. It contains the following Bash scripts:
- `./setup_gcp_infrastructure.sh ` for setting up a VM, pull a Git repository and run the `startup_script.sh`.
- `./delete_gcp_infrastructure.sh ` for deletting the infrastructure.
- `./upload_model_checkpoint.sh` to upload checkpoint files to a shared Google Cloud Storage.
- `./download_model_checkpoint.sh` to download checkpoint files from a shared Google Cloud Storage to the local instance.

Further details in [Link]

### API
Here how to use the API

## Demo
Here how to deploy and test the GUI
