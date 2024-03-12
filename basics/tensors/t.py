import torch

shape = (3,3,)
t1 = torch.rand(shape)
print(t1)
print(shape)

t2 = torch.ones(shape)

print(f"matmul {t1 @ t2}")
print(f"elemmul {t1.mul(t2)}")

t1_flattened = t1.flatten().type(torch.float32)
t2_flattened = t2.flatten().type(torch.float32)

dot_t1_t2 = t1_flattened.dot(t2_flattened)
elem_t1_t2 = t1_flattened.sum()

print(t1_flattened)
print(t2_flattened)
print(f"dot {dot_t1_t2}")
print(f"sum t1: {t1.sum()} t2: {t2.sum()}")


print(dot_t1_t2)
print(elem_t1_t2)
print(dot_t1_t2==elem_t1_t2)
print(dot_t1_t2.type())
print(elem_t1_t2.type())
