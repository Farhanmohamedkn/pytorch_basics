import torch
import numpy as np


# step 1: prediction(using linear regression)
# step 2: gradient computation
# step3: loss computation
# step 4: parameter updates

# Compute every step manually(Here we replace the manually computed gradient with autograd)
#means we use pytorch instead of numpy

# Linear regression
# f = w * x  

# here : f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # used  pytorchtensor instaed of numpy array
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)# x=1,y or f=2 ,   x=2, y or f=4 .....etc

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)# start our weights as zero as a torch auto grad function

# model output
def forward(x):
    return w * x

# loss = MSE in pytorch we can write manually as below
def loss(y, y_pred):
    return ((y_pred - y)**2).mean() #this is the equation for loss

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)


# def gradient(x, y, y_pred):
#     return np.mean(2*x*(y_pred - y)) #we calculated the gradient manually dx/dy

# **----- WE REPLACE THE ABOVE MANUALLY WRITTEN GRADIENT FUNCTION WITH PYTORCH AUTOGRAD FUNCTION -----*


print(f'Prediction before training: f(5) = {forward(5):.3f}') #so if x is 5 then f should be 10 but before training its zero beacuse of our weights

#after training we get 10 or close to 10 our actual vallue for function f(5) bcz our formula is 2X

#lets do training

# Training
learning_rate = 0.01
n_iters = 200

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)
    
   
   
    # calculate gradients
    # dw = gradient(X, Y, y_pred) we dont need it anymore

    
    
    # calculate gradients = backward pass pytorch
    l.backward()


    
    # update weights
    #w.data = w.data - learning_rate * w.grad
    with torch.no_grad():
        w -= learning_rate * w.grad

    
    
    # zero the gradients after updating
    w.grad.zero_()

    if epoch % 10 == 0: # so we can see the weight and loss in each 10 episode or epoch
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
     
print(f'Prediction after training: f(5) = {forward(5):.3f}') # see our prediction after training is 10 
# if you observe the output u can see that theweights are increase gradually and loss is reducing
# means slope is increasing thus more learning (dx/dy) 
