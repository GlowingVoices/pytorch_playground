
import torch


#define a shape
shape = (10,)

#create random tensors t1 and t2
t1 = torch.rand(shape).flatten().type(torch.float32)
t2 = torch.rand(shape).flatten().type(torch.float32)

print(t1)
print(t2)

#calculate the dot product of t1 and t2
dot = t1.dot(t2)

#calculate the elementwise multiple of t1 and t2
elem = t1.mul(t2)

#calculate the sum of t1 and t2
elem = elem.sum()

#print the dot product
print(dot)

#print the sum of the elementwise multiple
print(elem)

#print if they are the same
test = "the same" if dot-elem==0 else "not the same"
print(f"The sums of t1 and t2 are {test} value.")

#this should always be TRUE (they will be the same)
#BUT torch messes up at times and says they are not the same bc of floating point errors
#the dot product is the sum of the elementwise multiplication of two 1D vectors.
