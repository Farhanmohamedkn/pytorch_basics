import torch
import numpy as np


# Auto grad package and how we can use it to calculate
#gradient

x=torch.randn(3,requires_grad=True) # set some random tensor and we calculate the gradient of it
print(x)

#consider a function below
y= x+2 # pytorch will generate us a computational graph  and a gradient function so we can easily calculate gradient

print(y) #output= tensor([1.2547, 1.9429, 0.8709], grad_fn=<AddBackward0>) #here its AddBackward function generated by the pytorch bcz our operation is +(add)
# then we do the back propagation later

z=y*y*2 # lets consider another operation multiplication
z=z.mean() # then take the mean some random operations 

print(z) # our gradient function is now grad_fn=<MeanBackward0

#now if we want to calculate the gradient do the following
z.backward() #dz/dx
#now x has a grad attribute
print(x.grad) #if we make requires_grad=False then we get RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn


# rad is always store the values has memeory so when we do next iteration it must clear it for next epoch

#Lets do an experiment on a model
weights=torch.ones(4,requires_grad=True)
for epoch in range(3):
    model_output=(weights*3).sum()
    model_output.backward() #calculate gradient
    print(weights.grad) #output is not constant it added to the previous one due to memory
    weights.grad.zero_() # we reset the grad to zero so not get added for every epochs


# example with optimizer below
    
weights=torch.ones(4,requires_grad=True)
optimizer=torch.optim.SGD(weights,lr=0.1)
optimizer.step()
optimizer.zero_grad()