import torch
import numpy as np





#Lets create tensors of different dimensions
x=torch.empty(1) # A scalar #Lets create an empty tensor
print("1D=",x)

y=torch.empty(1,2) #Lets create an empty tensor of 2D
print("2D=",y)

z=torch.empty(1,2,3) #Lets create an empty tensor of 3D
print("3D=",z)

p=torch.empty(1,2,3,4) #Lets create an empty tensor of 4D
print("4D=",p)



# Lets create tensor filled with random values


#Lets create tensors of different dimensions
x=torch.rand(1) # A scalar #Lets create an random tensor
print("1D=",x)

y=torch.rand(1,2) #Lets create an random tensor of 2D
print("2D=",y)

z=torch.rand(1,2,3) #Lets create an random tensor of 3D
print("3D=",z)

p=torch.rand(1,2,3,4) #Lets create an random tensor of 4D
print("4D=",p)

# Lets put zeors in a tensor

x=torch.zeros(2,3) #Lets create an zero tensor of 2D
print("zero_2D=",x)

# Lets put ones in a tensor

x=torch.ones(2,3) #Lets create an zero tensor of 2D
print("ones_2D=",x)


# check the data type of tensor we created

x=torch.zeros(2,3) #Lets create an zero tensor of 2D
print("zero_2D data type =",x.dtype) # torch.float32

# lets assign a data type "int" to tensor
x=torch.ones(2,3,dtype=torch.int) #Lets create an one tensor of 2D
print("zero_2D data type =",x.dtype) # torch.int32



# lets check the size tensor
x=torch.ones(2,3,dtype=torch.int) #Lets create an one tensor of 2D
print("zero_2D size =",x.size()) # # lets assign a data type "int" to tensor
x=torch.ones(2,3,dtype=torch.int) #Lets create an one tensor of 2D
print("zero_2D data type =",x.dtype) # torch.Size([2, 3])


# Lets create a tensor from a random list or data

x=torch.tensor([2.5,0.1])
print("tensor from list=",x)

# Lets do basic mathematics operation using tensor and pytorch

x=torch.rand(2,2)
y=torch.rand(2,2)

#addition
z=x+y 
#or
z=torch.add(x,y)
# or we can inplace it by addition 
y.add_(x) # this will replace y value with new added value

# using the underscore we can inplace the tensor or replace the tensor
# eg "".mul_ ""



# Lets get the rows and coloms seperatly from a tensor
#called slicing operation

x=torch.rand(5,3)
print("whole tensor=",x)
print("first coloumn",x[:,0]) #get the zero coloumn
print("first row",x[0,:]) #get the zeroth row
print("get element at position 1,1 as tensor=",x[0,0]) #get the zeroth row

#if zou have only 1 element inside the tensor means scalar
#we can get the item as follows
print("get element at position 1,1 as avlue=",x[0,0].item())



# Lets reshape the tensor of (4,4) to any dimension we want

x=torch.rand(4,4)
print(x)
y=x.view(-1,8) #(used -1 so it automatically decide the dimension)
print("converted shape=",y)


#Lets convert tensor to numpy

a=torch.ones(5)
print(a)
b=a.numpy()
print ("converted numpy array=",b)

#lets convert numpy to tensor
a=np.ones(5)
print(a)
b=torch.from_numpy(a)
print ("converted tensor=",b)


# we can create and put the tensor in GPU if cuda is available
if torch.cuda.is_available():
    device=torch.device("cuda")
    x=torch.ones(5,device=device)
    y=torch.ones(5)
    z=x+y #now this operation will run in gpu much faster

# important we cant do this in numpy because
    #numpy can only handle Cpp
    #so if u wnat t o calculate numpy and tensor we have to
    #convert the tensor it back to cpu as follows

z=z.to("cpu") #now we can use numpy to manipualte it


# requires_grad= True :its default false but make it TRUE 
# so we can optimize the purticular variable later

x=torch.ones(5,requires_grad=True)
print(x)
