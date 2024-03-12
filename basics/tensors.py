import torch
import numpy as np


data = [[1,2],[3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
print(f"Ones tensor \n {x_ones}")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor}")
print(f"Ones Tensor: \n {ones_tensor}")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor {tensor.dtype}")
print(f"Device tensor is stored on {tensor.device}")

# tensors are created - by default - on the cpu.
# This moves them to the gpu if the gpu is available
# The transfer process can be expensive for larger tensors
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4,4)
print(f"First Row: {tensor[0]}")
print(f"First Column: {tensor[:,0]}")
print(f"Last Column: {tensor[...,-1]}")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)


#bridge with numpy

shape = (2,1)
t_zero = torch.zeros(shape)

print(t_zero)
n = t_zero.numpy()
print(f"n: {n}")
t_zero.add_(1)
print(f"n updated numpy: the numpy changes when the torch does: {n}")

new_t = torch.from_numpy(n)

print("testing if changing og torch (t_zero) alters new torch (new_t)")
print(new_t)
t_zero = t_zero.add_(1)
print(new_t)

#okay so it does
#This is IMPORTANT. How do we duplicate a numpy array without being dependent on the original?

print(t1)
t_new = t1
t1 = t1.subtract_(2)
print(t_new) #nope t_new is still dependent


#we can use t_new = t_old.clone().detach() to create a duplicate
#https://pytorch.org/docs/stable/tensors.html#torch.Tensor.clone
#https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor

#using detach().clone() gets into the logic of the computational graph
#(definitely fun, but not something to do until I have a decent grasp of the fundamentals of pytorch)


"""
t_next = t_new.clone().detach()
print(f"We just detacted and cloned from t_new: {t_next}")
print("We can now edit t_new without altering t_next & vice_versa")

print(f"t_new: {t_new}")
print(f"t_next{t_next}")

t_new = t_new.multiply_(4)
print(f"edited t_new: {t_new}")
print(f"unaltered t_next {t_next}")

print("That works! We can also edit t_next without altering t_new")
print(f"t_new {t_new}")
print(f"t_next {t_next}")

t_next = t_next.add(1)
print(f"Altered t_next {t_next}")
print(f"unaltered t_new {t_new}")

"""
