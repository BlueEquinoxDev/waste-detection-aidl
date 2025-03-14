# Waste Segmentation and detection AIDL project

This repository contains the code developed by Martí Fabregat, Rafel Febrer, Ferran Miró-Gea and Miguel Ortiz in the scope of AIDL postgraduate course in UPC (Universitat Politècnica de Catalunya). With supervision of Amanda Duarte.

Several models for instance segmentation and image classification have been trained and evaluated to segment and classify waste.

To do Project goal --> Marti

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
python -m scripts.download
```

In case of image classification the taco dataset is divided in its annotations and the background is removed, so that the waste is surrounded by back pixels.
This allows to use the Taco Dataset as a waste image classification dataset.

For image segmentation the Taco dataset is used "as it is".

The categories that will be used for classification and segmentation depend on the subversion of Taco Dataset selected.

* **Taco28** - Contains the complete Taco Dataset using the 28 supercategories as labels.
* **Taco5** - Contains a subsample of the images of the original Taco Dataset. (5 labels only). It is an "easier" task.
* **Taco1** - Contains the complete Taco Dataset using only 1 category as label ("waste").

#### Viola77

The [Viola77](https://huggingface.co/datasets/viola77data/recycling-dataset) dataset is used as well for classification. Under Apache 2.0 License.

#### Combination of Datasets

A combination of Taco and Viola77 Datasets have been created to increase the number of images for image classification tasks to test models under different situations.

* **taco39viola11** - Contains a Taco Dataset subsection of annotations that match the Viola categories plus the Viola Dataset, so it is the Dataset with the biggest number of images for classification.

### Exploratory data analysis

Explore the notebook ``demo.ipynb``, modified version of the original notebook from the [TACO Repository](https://github.com/pedropro/TACO) that inspects the dataset.
The dataset is in COCO format. It contains the source pictures, anotations and labels. For more details related with the datasource please refer to [TACO Repository](https://github.com/pedropro/TACO).

To do: Notebook for Viola --> Martí

## Training

### Image Classification with ResNet-50 --> to do Marti
#### Split dataset
To split the annotations for training and evaluation on **ResNet-50** use ``split_dataset.py`` according to this explanation. It has several optional flags.
```
python -m scripts.split_dataset --dataset_dir=data --dataset_type=classification [--test_percentage=0.1] [--val_percentage=0.1] [--seed=123] [--verbose=False] 
```
* Indicate the annotations directory using ``--dataset_dir``.
* Indicate the dataset to use with ``--dataset_type`` flag. In case of classification use:
    * ``--dataset_type=classification``
    (Use it to run classification in taco28, taco5 or taco30viola11).
##### Optional:
* Use ``--test_percentage`` if you want to use a test split different than default 0.1 (10%).
* Use ``--val_percentage`` if you want to use a validation split different than default 0.1 (10%).
* Use ``--seed`` if you want to have a different random output. Default 123.
* Use ``--verbose`` (bool) if you want to have printed text on the console during execution.
#### Dataset classes
##### ResNet-50 for Viola77
##### ResNet-50 for Taco
#### Train
#### Evaluate
#### Results

### Image Classification with ViT --> To do Ferran
#### Split dataset
To split the annotations for training and evaluation on **ViT** use ``split_dataset.py`` following the same procedure as in **ResNet-50**.
```
python -m scripts.split_dataset --dataset_dir=data --dataset_type=classification [--test_percentage=0.1] [--val_percentage=0.1] [--seed=123] [--verbose=False] 
```

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

### Instance segmentation with Mask R-CNN --> To do Miquel
#### Split dataset
To split the annotations for training and evaluation in **Mask R-CNN** use ``split_dataset.py``.

```
python -m scripts.split_dataset --dataset_dir=data --dataset_type=taco1 [--test_percentage=0.1] [--val_percentage=0.1] [--seed=123] [--verbose=False]
```
* Indicate the annotations directory using ``--dataset_dir``.
* Indicate the dataset to use with ``--dataset_type`` flag. It depends on the task to do. It can be:
    * ``taco28`` for **instance segmentation** in *taco28* dataset (Taco dataset with 28 categories, includes all data)
    * ``taco5`` for **instance segmentation** in *taco5* dataset (Taco dataset with a subsample of 5 categories)
    * ``taco1`` for **instance segmentation** in *taco1* dataset (Taco dataset only segmenting waste from background, includes all data)
##### Optional:
* Use ``--test_percentage`` if you want to use a test split different than default 0.1 (10%).
* Use ``--val_percentage`` if you want to use a validation split different than default 0.1 (10%).
* Use ``--seed`` if you want to have a different random output. Default 123.
* Use ``--verbose`` (bool) if you want to have printed text on the console during execution.

#### Dataset classes
The Taco Dataset for mask R-CNN class in ``custom_datasets/taco_dataset_mask_r_cnn_update.py`` has the functionality to load the Taco Dataset in for Instance Segmentation.

#### Train
Run ``python -m scripts.train_mask_r_cnn``

#### Evaluate
Run ``python -m scripts.test_mask_r_cnn``

#### Results

### Instance segmentation with Mask2Former --> To do Rafa
#### Split dataset
To split the annotations for training and evaluation on **Mask2Former** use ``split_dataset.py`` following the same procedure as in **Mask R-CNN**.

```
python -m scripts.split_dataset --dataset_dir=data --dataset_type=taco1 [--test_percentage=0.1] [--val_percentage=0.1] [--seed=123] [--verbose=False]
```

#### Dataset classes

#### Train
Run ``python -m scripts.train_mask2former_segmentation``

#### Evaluate
Run ``python -m scripts.test_mask2former_segmentation``

#### Results


## MLOps

### Running the application with Docker --> To do Ferran
Build the image with:
```docker build -t waste-detection-app .```

Run specific Python file:
```docker run --rm waste-detection-app <FILE_NAME.py>```

### Google Cloud --> To do Ferran
This repository automates the setup of the GCP infrastructure. It contains the following Bash scripts:
- `./setup_gcp_infrastructure.sh ` for setting up a VM, pull a Git repository and run the `startup_script.sh`.
- `./delete_gcp_infrastructure.sh ` for deletting the infrastructure.
- `./upload_model_checkpoint.sh` to upload checkpoint files to a shared Google Cloud Storage.
- `./download_model_checkpoint.sh` to download checkpoint files from a shared Google Cloud Storage to the local instance.

Further details in [Link]

### API --> To do Rafa
Here how to use the API

## Demo
Here how to deploy and test the GUI

## Next Steps --> To do Marti

