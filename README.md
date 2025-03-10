# Handwritten Mathematical Symbol Classifier

This project is a CNN-based model for classifying handwritten mathematical symbols. The dataset consists of grayscale images of various symbols, and the model is trained using PyTorch.

## Setup
### 1. Install Dependencies
```bash
pip install torch torchvision matplotlib pandas scikit-learn
```

### 2. Download Dataset
If the dataset is from a Kaggle competition, use:
```bash
kaggle competitions download -c torch-it-up
```
Make sure to set up the Kaggle API first.

### 3. Extract Dataset
```python
import zipfile
import os

zip_path = "dataset.zip"  # Replace with actual path
extract_path = "dataset"

os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

## Training the Model
Run the training script to train the model.
```python
python train.py
```
This will train the CNN on the dataset and save the model.

## Predicting a Single Image
To predict a single image, use:
```python
python predict_single.py --image_path path/to/image.png --model_path model.pth
```
This will display the image and output the predicted class.

## Checking Accuracy
To evaluate the model on a validation set:
```python
python evaluate.py
```
This will print the training and validation accuracy.

## Submission
To generate predictions for the test set:
```python
python generate_submission.py
```
This will create a `submission.csv` file in the correct format for the competition.

## Notes
- Ensure that the dataset paths are correctly set.
- Use GPU for faster training if available.
- Modify the CNN architecture if better accuracy is required.

This repository contains all necessary scripts to train, evaluate, and test the model efficiently.

