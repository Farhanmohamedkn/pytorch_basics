


# step 1: prediction(using linear regression) (using pytorch) 
# step 2: gradient computation (using pytorch)
# step3: loss computation(using pytorch)
# step 4: parameter updates(using pytorch)

# Compute every step using pytorch 

# design a model(input size, output size,forward pass)
# construct loss and optimizer
#training loop:
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weights




import torch
import torch.nn as nn

# Linear regression
# f = w * x  

# here : f = 2 * x


# X = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # used  pytorchtensor instaed of numpy array
# Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)# x=1,y or f=2 ,   x=2, y or f=4 .....etc

# 0) Training samples, watch the shape!
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32) # our shape of the tesnor is also changed becz we using nn.linear model we have tp pass a tensor
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32) # not a float value ,it require a 2D shape or list shape as input and output

n_samples, n_features = X.shape

print(f'#samples: {n_samples}, #features: {n_features}')

# 0) create a test sample tensor see not a float not a simple number
X_test = torch.tensor([5], dtype=torch.float32)


# 1) Design Model: Weights to optimize and forward function

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)# start our weights as zero as a torch auto grad function

# # model output
# def forward(x):
#     return w * x

# # loss = MSE in pytorch we can write manually as below
# def loss(y, y_pred):
#     return ((y_pred - y)**2).mean() #this is the equation for loss

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)


# def gradient(x, y, y_pred):
#     return np.mean(2*x*(y_pred - y)) #we calculated the gradient manually dx/dy
# print(f'Prediction before training: f(5) = {forward(5):.3f}') #so if x is 5 then f should be 10 but before training its zero beacuse of our weights


# **----- WE REPLACE THE ABOVE MANUALLY WRITTEN fORWARD GRADIENT and LOSS function WITH PYTORCH  functionS -----****
# **-----Removed weights also utherwise because then our pytorch model know the parameters-------***



# 1) Design Model, the model has to implement the forward pass!

# Here we can use a built-in model from PyTorch
input_size = n_features
output_size = n_features

# we can call this model with samples X
model = nn.Linear(input_size, output_size)

'''
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
'''

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}') # see here we passsed a tensor to the model not a float value 5
#where X_test = torch.tensor([5], dtype=torch.float32) is a tensor



# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 100

# callable function
loss = nn.MSELoss()  # usedpytorches mse loss function

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # put model.parametr as the values to optimized

#after training we get 10 or close to 10 our actual vallue for function f(5) bcz our formula is 2X

#lets do training

# Training
learning_rate = 0.01
n_iters = 200 #no of epochs

# for epoch in range(n_iters):
#     # predict = forward pass
#     y_pred = forward(X) #predicted y

#     # loss
#     l = loss(Y, y_pred)
    
   
   
#     # calculate gradients
#     # dw = gradient(X, Y, y_pred) we dont need it anymore

    
    
#     # calculate gradients = backward pass pytorch
#     l.backward() #this will calculate the slope for us


    
#     # update weights
#     #w.data = w.data - learning_rate * w.grad
#     with torch.no_grad():
#         w -= learning_rate * w.grad #this part should not be a part of our computational graph so we use "with no grad" function to clear menory each time

    
    
#     # zero the gradients after updating
#     w.grad.zero_() #if we don use it we get a value more than which is obviosly wrong becs our formula is 2x 
#     # so basically to clear the w.grad attribute so our gradients are zero again




#     if epoch % 10 == 0: # so we can see the weight and loss in each 10 episode or epoch
#         print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
 # **----- WE REPLACE THE ABOVE MANUALLY WRITTEN training step WITH pytorch -----*    

# 3) Training loop
for epoch in range(n_iters):
    # predict = forward pass
    # y_predicted = forward(X) #predicted y manually written

    y_predicted = model(X) #model created using pytorch linear regresion function
    # loss
    l = loss(Y, y_predicted) #calculate loss

    # calculate gradients = backward pass
    l.backward() #calculated slope

    # update weights
    optimizer.step() #we dont need to manually update the weight anymore optimzer will do it

    # zero the gradients after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        # print('epoch ', epoch+1, ': w = ', w, ' loss = ', l)
        [w, b] = model.parameters() # unpack parameters
        #w is weight its a liso list 2D we know
        #b is bias
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l) # u can see at the output it is tensors not float values

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}') 
     
# print(f'Prediction after training: f(5) = {forward(5).item():.3f}')# see our prediction after training is 10 
# if you observe the output u can see that theweights are increase gradually and loss is reducing
# means slope is increasing thus more learning (dx/dy) 
