import torch                        #pytorch library - for for tensor computations
import torch.nn as nn               #for building neural networks
import numpy as np                  #numerical data manipulation
from sklearn import datasets        #To generate and load datasets
import matplotlib.pyplot as plt     #for creating visualizations


# 0. prepare data 
x_numpy, y_numpy = datasets.make_regression(n_samples=70, n_features =1, noise=17, random_state=1)
# x_numpy: A 2D NumPy array with shape (n_samples, n_features)
# y_numpy: A 1D NumPy array with shape (n_samples,)


x = torch.from_numpy(x_numpy.astype(np.float32)) #convert data type for tensor computations
y = torch.from_numpy(y_numpy.astype(np.float32)) 
y = y.view(y.shape[0],1) # Reshape into a 2D tensor with shape (n, 1)

n_samples, n_features = x.shape

# 1. define model 
input_size = n_features 
output_size = 1 
model = nn.Linear(input_size, output_size) 

# 2. define loss function and optimizer 
criterion = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. train loops 
num_epochs = 100 
for epoch in range(num_epochs):
    #forward pass and loss 
    y_predicted = model(x) 
    loss = criterion(y_predicted, y)

    #backward pass 
    loss.backward() 

    #update 
    optimizer.step() 

    optimizer.zero_grad() 

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/ {num_epochs}], Loss: {loss.item():0.4}')


#plot 
predicted = model(x).detach().numpy() 
plt.plot(x_numpy, y_numpy, 'ro', label='Original data') 
plt.plot(x_numpy, predicted, 'b',label='Fitted line') #create line plot
plt.show()