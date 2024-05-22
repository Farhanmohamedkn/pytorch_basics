import torch
import numpy as np


# back propagation is an algorithm (based on chain rule) dz/dx= (dz/dy) * (dy/dx)
#step 1: Forward pass: Compute Loss (where we apply all the functions and compute loss)
#step2: Compute local gradient at that node
#step3: Backward pass: Compute the "gradient of Loss/ gradient of Weights "using the chain rule


# lets do back propagation

x=torch.tensor(1.0) #a float tensor defined
print("x value=",x)
y=torch.tensor(2.0) #a float tensor defined
print("y value=",y)
w=torch.tensor(1.0,requires_grad=True) # we need to calculate the gradient of the weight so it is true

#lets do step 1 forward pass and compute loss

y_temp=w*x # our function

loss=(y_temp-y)**2
print("computed loss",loss) # u can see at the out pu its powerbackward

#step2 and step3 together
#backward pass

loss.backward()
print("weight grad",w.grad) # now our weight has the grad attribute