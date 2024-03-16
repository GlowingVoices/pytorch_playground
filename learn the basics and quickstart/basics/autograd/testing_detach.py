import torch


shape = (5,3,)

x = torch.rand(shape)
x.requires_grad = True
print(f"A random tensor sampled with rand (uniform distribution) {x}")
y = x
print(f"We just set {y} equal to x")
print(y)

print(f"We're updating x, this changes y")
x = x.add_(1)
print(f"The x-update-affected y: {y}")

z = x.clone().detach()
print(f"Our detached clone of x: {z}")
print("It's independent of x.")

print(x.requires_grad)
print(z.requires_grad)
