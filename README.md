# Stroke Prediction using Clinical and Social Features in Machine Learning

## Overview
Implementation of machine learning models for stroke prediction using healthcare data. Models include Logistic Regression, Dense Neural Network, and Convolutional Neural Network.


## Dataset
Healthcare stroke prediction dataset with 5110 samples
Features include:
- Demographics (age, gender)
- Medical history (hypertension, heart disease)
- Lifestyle factors (smoking status, BMI)
- Clinical measurements (glucose levels)

## Requirements
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import random
import csv
import datetime
import time
import pandas as pd
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

```

## Usage
Run each cell separately to initialize, train, and test models. 

## Results
Models save metrics and visualizations to `model_outputs/`:
- Model parameters
- Training / Testing metrics graphs
- Model performance comparisons

## Authors
Aidan Chadha
