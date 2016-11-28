import random
import math
from range_algorithm import *
from scipy.optimize import minimize


class SvmAlgorithm(RangeAlgorithm):

    def __init__(self, fig, NumbKer, NumbSubPlot, KernelType):
        super().__init__(fig, NumbKer, NumbSubPlot)
        # type of Vectors kernel
        self.KernelType = KernelType
        # parameter
        self.C = 10
        #minimal error
        self.eps = 1e-05

    def __del__(self):
        super().__del__()

    def draw_dot(self, i, r):
        return super().draw_dot(i, r)

    def init_arrays(self):
        # NDxND
        self.KerArr = np.zeros((self.ND, self.ND), dtype=float)
        # x_i its line!   NumbKer lines, ND columns
        self.X = np.zeros((self.ND, 2), dtype=float)
        # y_i its row
        self.Y = np.zeros((self.ND,), dtype=float)
        # x0 for func, row
        self.L0 = np.zeros((self.ND,), dtype=float)
        #Weight array
        self.Weight = np.zeros((2,), dtype = float)
        self.Weight0 = 0

    def build_image(self):
        super().build_image()
        self.pix.set_title('SVM classification')

    def calculate_weight(self):
        self.init_arrays()
        # fill X and Y massives
        for i in range(0, self.ND):
            self.Y[i] = self.Dots[i][3]
            self.X[i, 0] = self.Dots[i][0] / 100
            self.X[i, 1] = self.Dots[i][1] / 100
        for i in range(0, self.ND):
            for j in range(0, self.ND):
                self.KerArr[i][j] = self.KernelType(self.X[i][:],
                                                     self.X[j][:])

        print(self.KerArr)

        Lang = lambda Lamb: -1.0 * Lamb.sum() + 0.5 * np.transpose(self.Y * Lamb).dot(self.KerArr.dot(self.Y * Lamb))

        cons = ({'type': 'eq', 'fun': lambda Lamb: np.sum(self.Y * Lamb)})
        opt = {'disp': True, 'maxiter': 1000, 'ftol': 1e-06}
        bound = tuple([(0, self.C)] * self.ND)

        res = minimize(Lang, self.L0, method='SLSQP', bounds=bound,
                       jac=False, constraints=cons, options=opt)
        
        self.Optim = res.x
        np.set_printoptions(suppress=True)
        print('optim array: {}'.format(self.Optim))

        self.Weight = (self.Y * self.Optim).dot(self.X)
        np.set_printoptions(suppress=True)
        print('weight array: {}'.format(self.Weight))

        for i in range(0, self.ND):
            if self.eps < self.Optim[i] < self.C - self.eps:
                self.Weight0 = np.dot(self.Weight, self.X[i][:]) - self.Y[i]
                break
        print('Weight0 = {}'.format(self.Weight0))

    def calculate_pixel(self):
        for self.px in range(0, 100):
            if (self.px % 25 == 0):
                print('{0} iterate'.format(self.px))
            for self.py in range(0, 100):
                VecArr = np.array([[self.px / 100], [self.py / 100]])
                Class = 0
                for i in range(0, self.ND):
                    Class += self.Y[i] * self.Optim[i] * self.KernelType(self.X[i][:], VecArr)
                Class -= self.Weight0
#                if -1 < np.dot(self.Weight,VecArr) - self.Weight0 < 1:
#                    self.ArrayZ[self.px, self.py] = np.sign(Class) / 2
#                else:
                self.ArrayZ[self.px, self.py] = Class

    def Main(self):
        self.ND = len(self.Dots)
        self.build_image()
        print('Starting SVM algorithm')
        self.calculate_weight()
        self.calculate_pixel()
        for i in range(0, self.ND):
            if abs(self.C - self.Optim[i]) < self.eps:
#               support wrong
                self.draw_dot(i, 25)
            elif self.Optim[i] > self.eps:
#               support vector
                self.draw_dot(i, 100)
            else:
#               non-informative
                self.draw_dot(i, 50)
        self.im.set_data(self.ArrayX, self.ArrayY, self.ArrayZ)
        self.pix.images.append(self.im)
        self.cb = plt.colorbar(self.im)
