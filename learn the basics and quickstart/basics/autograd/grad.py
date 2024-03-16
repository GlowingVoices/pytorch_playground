import torch
from torch import nn

x = torch.ones(5) #input tensor

y = torch.zeros(3) #expected output
w = torch.randn(5,3,requires_grad=True) #random weight vector
b = torch.randn(3, requires_grad=True) #random bias vector

print(x.shape)
print(w.shape)
print(b.shape)
print(x)
print(w)
z = torch.matmul(x,w)+b #linear layer
#binary crossentropy loss calculation
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)

#defining z as the matrix multiple of x,w + b
#this is functionally a dense layer (linear combination)
z = torch.matmul(x,w) + b
print(z)


# I'm redefining it in a no_gradient state.
with torch.no_grad():
    z= torch.matmul(x,w) +b
print(z.requires_grad)

inp = torch.eye(4,5, requires_grad=True)

out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph = True)

print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph = True)


print(f"Second call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph = True)

print(f"\nCall after zeroing gradients\n{inp.grad}")
