import torch

x = torch.ones(5) #input tensor

y = torch.zeros(3) #expected output
w = torch.randn(5,3,requires_grad=True) #random weight vector
b = torch.randn(3, requires_grad=True) #random bias vector
z = torch.matmul(x,w)+b #linear layer
#binary crossentropy loss calculation
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
