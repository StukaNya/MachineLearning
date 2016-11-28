from init_dots import *
from range_algorithm import *
from svm_algorithm import *


# main script
# lambda functions of Kernel
linear_kernel = lambda x1, x2: np.dot(x1, x2)
polynom_kernel = lambda x1, x2, p = 3: (1 + np.dot(x1, x2)) ** p
gaussian_kernel = lambda x1, x2, sigma = 5.0: np.exp( - np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

fig = plt.figure()

# create dots with:           Numb  X   Y   Sigma  Colour
BlueDots = NormVariateDots(fig, 15, 35, 35, 20, "blue")
RedDots = NormVariateDots(fig, 15, 65, 65, 20, "red")

#                     sizeKernel\place of plot\func kernel
DoLearn = RangeAlgorithm(fig, 3, 122)
DoSVM = SvmAlgorithm(fig, 3, 121, polynom_kernel)

Draws = [BlueDots, RedDots, DoSVM]

for Draw in Draws:
    Draw.Main()

plt.show()
