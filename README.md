# Linear Regression with PyTorch

## Introduction
This program demonstrates the implementation of a simple linear regression model using PyTorch. It uses a generated dataset with sklearn to predict a dependent variable (y) based on an independent variable (x). The model is trained using gradient descent to minimize the mean squared error (MSE).

### Libraries to import: 
```
import torch                        #pytorch library - for tensor computations
import torch.nn as nn               #for building neural networks
import numpy as np                  #numerical data manipulation
from sklearn import datasets        #To generate and load datasets
import matplotlib.pyplot as plt     #for creating visualizations
```

### Generating dataset: 
```
x_numpy, y_numpy = datasets.make_regression(n_samples=70, n_features =1, noise=17, random_state=1)
# x_numpy: A 2D NumPy array with shape (n_samples, n_features)
# y_numpy: A 1D NumPy array with shape (n_samples,)
```
n_fueatures= 1 - Represent 2D structure

![Figure_1](https://github.com/user-attachments/assets/f3096346-bb31-494b-befd-be15e8deac46)
