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

As can be seen taco is a unbalanced dataset for waste image segmentation:
![taco28_dataset](readme_images/taco28.png)

#### Viola77

The [Viola77](https://huggingface.co/datasets/viola77data/recycling-dataset) dataset is used as well for classification. Under Apache 2.0 License.

Viola is a close to perfectly balanced dataset:
![Viola_dataset](readme_images/viola77.png)

#### Combination of Datasets

A combination of Taco and Viola77 Datasets have been created to increase the number of images for image classification tasks to test models under different situations.

* **taco39viola11** - Contains a Taco Dataset subsection of annotations that match the Viola categories plus the Viola Dataset, so it is the Dataset with the biggest number of images for classification.

![taco_and_viola](readme_images/taco39viola11.png)

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

This repository provides an implementation of ResNet-50 for waste classification using the Viola77 dataset. It includes scripts for hyperparameter optimization using Optuna, model training, and evaluation. 

##### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install torch torchvision transformers optuna numpy pandas matplotlib seaborn scikit-learn tqdm datasets
```

#### Scripts Overview

##### 1. `viola_dataset_resnet.py`
- **Description:** Custom PyTorch dataset class for the Viola77 dataset.
- **Inputs:**
  - Hugging Face dataset
- **Outputs:**
  - Preprocessed dataset for training and testing

##### 2. `optuna_resnet_hparams.py`
- **Description:** Uses Optuna to find the best hyperparameters for training the ResNet-50 model.
- **Inputs:**
  - Search space includes learning rate, dropout rate and optimizer type.
- **Outputs:**
  - Saves the best hyperparameters in `hparams.json`.

##### 3. `train_resnet_classification.py`
- **Description:** Trains a ResNet-50 model on the Viola77 dataset.
- **Inputs:**
  - Dataset from Hugging Face (`viola77data/recycling-dataset`)
  - If `enhanced_hparams=True`, loads hyperparameters from `hparams.json`
- **Outputs:**
  - Saves the trained model as `best_resnet50.pth`
  - Generates training metrics and confusion matrices

##### 3.1. Modification:
Added `enhanced_hparams` flag. If `True`, executes `optuna_resnet_hparams.py` first to determine the best hyperparameters before training.

##### 4. `test_resnet_classification.py`
- **Description:** Evaluates the trained ResNet-50 model on the test dataset.
- **Inputs:**
  - Loads `best_resnet50.pth`
  - Uses the Viola77 test dataset
- **Outputs:**
  - Accuracy, classification report, confusion matrix

#### How to Run

1. **Train the Model**
- Without optimized hyperparameters:
```bash
python -m scripts.train_resnet_classification
```
- With optimized hyperparameters:
```bash
python -m scripts.train_resnet_classification.py --enhanced_hparams True
```

2. **Test the Model**
```bash
python -m scripts.test_resnet_classification.py
```

#### Results

Include plots and metrics here:

- **Training Loss & Accuracy:**
  ![image](https://github.com/user-attachments/assets/343e2ff4-2995-4a68-b97c-908be6126b8e)

- **Confusion Matrix (Train & Test):**
  - **Train**:
    ![image](https://github.com/user-attachments/assets/0b24da25-9438-453d-af6e-f8c1e68b8e06)

  - **Test**:
    ![image](https://github.com/user-attachments/assets/9d5a8e1e-bffe-41a5-9454-438db90aa6f7)

- **Evaluation Histogram**:
  ![image](https://github.com/user-attachments/assets/2e889327-4b91-4f3f-a6ef-ce6375c5285e)
    
- **Classification Report:**
```
                   precision    recall  f1-score   support

        aluminium       1.00      0.96      0.98        25
        batteries       0.92      1.00      0.96        24
        cardboard       0.97      1.00      0.98        28
  
disposable plates       0.96      1.00      0.98        26
            glass       1.00      0.97      0.98        29
     hard plastic       0.90      0.93      0.91        28
            paper       0.95      0.95      0.95        22
      paper towel       1.00      1.00      1.00        40
      polystyrene       1.00      0.94      0.97        36
    soft plastics       0.92      0.89      0.91        27
    takeaway cups       1.00      1.00      1.00        26

         accuracy                           0.97       311
        macro avg       0.97      0.97      0.97       311
     weighted avg       0.97      0.97      0.97       311
```
     
- **Overall Accuracy: 0.9678**


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

### Instance segmentation with Mask2Former
#### Split dataset
To split the annotations for training and evaluation on **Mask2Former** use ``split_dataset.py`` following the same procedure as in **Mask R-CNN**.

```
python -m scripts.split_dataset --dataset_dir=data --dataset_type=taco1 [--test_percentage=0.1] [--val_percentage=0.1] [--seed=123] [--verbose=False]
```

#### Dataset classes

#### Train
The mask2former model has been finetunned using the weights from ``facebook/mask2former-swin-tiny-ade-semantic``.

To train mask2former model in any of the datasets, do:
```
python -m scripts.train_mask2former_segmentation --dataset_type=taco1 [--batch_size=1] [--checkpoint_path=your_checkpoint_path]
```
Checkpoints will be saved in the results folder.

#### Evaluate
To evaluate the model in the test set of the dataset, do:
```
python -m scripts.test_mask2former_segmentation --checkpoint_path=your_checkpoint_path
```
parser.add_argument('--checkpoint_path', required=False, help='Checkpoint path', type=str, default="")

#### Results
In general Mask2former learns to segment well the background from the waste but fails mostly in classifying the waste.
Notice that, because of the unbalance of the dataset some classes may never apear in the test dataset and therefore the mIoU could be 0.

##### Taco1
- **Description**:
Training of the Mask2former using the taco1 dataset in the task of instance segmentation.

- **Outputs:**
Train of the Mask2former in taco1 for 12 epochs:
![taco1_mask2former](readme_images/train_mask2former_taco1.png)
Best model is found at epoch 10 and metrics are computed.

- **Metrics**:

Categories | mIoU
--- | ---
Background | 0.9860629439353943
Waste | 0.5686376094818115

##### Taco5
- **Description**:
Training of the Mask2former using the taco5 dataset in the task of instance segmentation.

- **Outputs:**
Train of the Mask2former in Taco5 for 20 epochs:
![taco5_mask2former](readme_images/train_mask2former_taco5.png)
Best model is found at epoch 20 and metrics are computed.

- **Metrics**:

Categories | mIoU
--- | ---
Background | 0.9841329455375671
Bottle | 0.2138008177280426
Carton | 0.06941558420658112
Cup | 0.06950430572032928
Can | 0.13832922279834747
Plastic film | 0.0

##### Taco28
- **Description**:
Training of the Mask2former using the taco28 dataset in the task of instance segmentation.

- **Outputs:**
Train of the Mask2former in Taco28 for 20 epochs:
![taco28_mask2former](readme_images/train_mask2former_taco28.png)
Best model is found at epoch 14 and metrics are computed.

- **Metrics**:

Categories | mIoU
--- | ---
Background | 0.9905351996421814
Aluminium foil | 0.0
Battery | 0.0
Blister pack | 0.0
Bottle | 0.11504539847373962
Bottle cap | 0.04049689695239067
Broken glass | 0.0
Can | 0.03863873332738876
Carton | 0.0264853797852993
Cigarette | 0.0204333309084177
Cup | 0.041645195335149765
Food waste: | 0.0
Glass jar | 0.0
Lid | 0.012108653783798218
Other plastic | 0.018098747357726097
Paper | 0.010264495387673378
Paper bag | 0.0
Plastic bag & wrapper | 0.13824139535427094
Plastic container | 0.0058823530562222
Plastic glooves | 0.0
Plastic utensils | 0.0038472835440188646
Pop tab | 0.0
Rope & strings | 0.0
Scrap metal | 0.0
Shoe | 0.0
Squeezable tube | 0.0
Straw | 0.008996364660561085
Styrofoam piece | 0.01385095901787281
Unlabeled litter | 0.0


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

### API
To evaluate images outside the dataset an API with WebApp has been developed.

#### API configuration:
Before launching the API you need to create a `.env` file in the project root folder.

The `.env` file contains 3 variables, for example:
```
FLASK_SECRET_KEY=secret_key
MODEL_NAME=MASK2FORMER
CHECKPOINT=checkpoint_epoch_7_mask_rcnn_taco1.pt
```
The `MODEL_NAME` can be `MASK2FORMER` or `MASK_R-CNN`. The API has not been abilitated yet for classification models.

The checkpoints of the models to test for the app should be placed in the folder `app/checkpoint`.
![api_folders](readme_images/API_folders.png)

### Running the API
To run the api in local and in debug mode, do:
```python -m app.app```
The flask app will be launched in your localhost.

## Demo
By using the developed API the user can detect waste using the API or by using the web app (GUI). In the following points is explained how to interact with.

### Test images with API
Make sure the API is running in your localhost. Then, use the example code provided in the file ``test_api.py`` to create you own request to the API.
Just run:
```python -m app.test_api```
The API will process the pictures sent and will return the detections and the images with the segmentation.
Check ``app/results``folder after running test_api.py to see the results.

### WebApp GUI
The WebApp allows to process images in a more user frendly approach.
To use the user interface open your prefered browser and connect to your localhost port 8000: http://localhost:8000.

If the API is running you should get to the home page:
![web_app_home](readme_images/webapp_home.png)
**Select a picture** and click on **Upload**.

You should see your picture in the web.
![upload_image](readme_images/upload_image.png)
Click on **Predict** to generate the segmentations.

The output will be different according to the model used in the API. Here an example:
![web_pred](readme_images/web_prediction.png)
To try another picture use the **Try a different image** button.

## Next Steps --> To do Marti
- Implement the GUI in 

