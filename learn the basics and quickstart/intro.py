import torch

z = torch.zeros(5,3)
print(z)
print(z.dtype)

i = torch.ones((5,3), dtype=torch.int16)
print(i)

torch.manual_seed(1729)
r1 = torch.rand(2,2)
print('A random tensor:')
print(r1)

r2 = torch.rand(2, 2)
print("A differennt random tensor")
print(r2)


torch.manual_seed(1729)
r3 = torch.rand(2,2)
print('The same tensor:')
print(r3)



ones = torch.ones(2,3)
print(ones)

twos = torch.ones(2,3) * 2
print(twos)

threes = ones + twos
print(threes)
print(threes.shape)

r1 = torch.rand(2,3)
r2 = torch.rand(3,2)
r3 = r1+r2
