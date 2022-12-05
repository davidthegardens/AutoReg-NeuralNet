import numpy as np
from numpy.random import randn
from math import sqrt

outp1=np.random.rand(3,5)
outp=np.random.randn(3,5)
std = sqrt(2.0 / 10)
print(outp1*std)
print(outp*std)

lower, upper = -(sqrt(6.0) / sqrt(4 + 6)), (sqrt(6.0) / sqrt(4 + 6))
# generate random numbers
numbers = np.array([100,600,-100,600])
# scale to the desired range
scaled = lower + numbers * (upper - lower)

print(scaled)

coeff=sqrt(6.0)/sqrt(4 + 6)
# generate random numbers
numbers = np.array([100,600,-100,600])
# scale to the desired range
scaled = (numbers *coeff*2)-coeff

print(scaled)


