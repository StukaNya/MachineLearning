from init_dots import *
from range_algorithm import *
from svm_algorithm import *


# main script
# lambda functions of Kernel
linear_kernel = lambda x1, x2: np.dot(x1, x2)
polynom_kernel = lambda x1, x2, p = 3: (1 + np.dot(x1, x2)) ** p
square_kernel = lambda x1, x2, p = 2: np.dot(x1,x2) ** p
neural_net_kernel = lambda x1, x2: np.arctan(0.5 * np.dot(x1, x2) + 0.5)
gaussian_kernel = lambda x1, x2, sigma = 5.0: np.exp( - np.linalg.norm((x1- x2)*100) ** 2 / (2 * (sigma ** 2)))

#define names of functions
linear_kernel.__name__ = 'Linear kernel'
polynom_kernel.__name__ = 'Polynom kernel'
square_kernel.__name__ = 'Square kernel'
neural_net_kernel.__name__ = 'NeuroNetwork kernel'
gaussian_kernel.__name__ = 'Gaussian kernel'

fig = plt.figure()

# create dots with:           Numb  X   Y   Sigma  Colour
BlueDots = NormVariateDots(fig, 15, 20, 20, 12, "blue")
RedDots = NormVariateDots(fig, 15, 55, 55, 25, "red")

#                     sizeKernel\place of plot\func kernel
DoLearn = RangeAlgorithm(fig, 3, 122)
DoSVM1 = SvmAlgorithm(fig, 3, 221, linear_kernel)
DoSVM2 = SvmAlgorithm(fig, 3, 222, polynom_kernel)
DoSVM3 = SvmAlgorithm(fig, 3, 223, square_kernel)
DoSVM4 = SvmAlgorithm(fig, 3, 224, neural_net_kernel)

Draws = [BlueDots, RedDots, DoSVM1, DoSVM2, DoSVM3, DoSVM4]

for Draw in Draws:
    Draw.Main()

plt.show()
