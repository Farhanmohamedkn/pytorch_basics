import torch
import numpy as np


# step 1: prediction(using linear regression)
# step 2: gradient computation
# step3: loss computation
# step 4: parameter updates

# Compute every step manually(means we dont use pytorch only numpy)

# Linear regression
# f = w * x  

# here : f = 2 * x (our function means if x=1 then f= 2)
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32) # x=1,y or f=2 ,   x=2, y or f=4 .....etc

w = 0.0 # start our weights as zero

# model output
def forward(x):
    return w * x

# loss = MSE in pytorch we can write manually as below
def loss(y, y_pred):
    return ((y_pred - y)**2).mean() #this is the equation for loss

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
def gradient(x, y, y_pred):
    return np.mean(2*x*(y_pred - y)) #we calculated the gradient manually dx/dy

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
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 2 == 0: # so we can see the weight and loss in each episode or epoch
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
     
print(f'Prediction after training: f(5) = {forward(5):.3f}') # see our prediction after training is 9.62 close to 10
# if you observe the output u can see that theweights are increase graduklly and loss is reducing
# means slope is increasing thus more learning (dx/dy) 
