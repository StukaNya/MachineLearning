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
        self.C = 100
        # number of errors
        self.LOOErr = []
        # minimal error
        self.eps = 1e-05

    def __del__(self):
        super().__del__()
        self.LOOErr.clear()

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

    def calculate_weight(self, display=True):
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

        Lang = lambda Lamb: -1.0 * Lamb.sum() + 0.5 * np.transpose(self.Y * Lamb).dot(self.KerArr.dot(self.Y * Lamb))

        cons = ({'type': 'eq', 'fun': lambda Lamb: np.sum(self.Y * Lamb)})
        opt = {'disp': display, 'maxiter': 50, 'ftol': 1e-06}
        bound = tuple([(0, self.C)] * self.ND)

        res = minimize(Lang, self.L0, method='SLSQP', bounds=bound,
                       jac=False, constraints=cons, options=opt)
        
        self.Optim = res.x
        if display == True:
            np.set_printoptions(suppress=True)
            print('optim sign array: {}'.format(self.Optim * self.Y))

        self.Weight = (self.Y * self.Optim).dot(self.X)
        if display == True:
            np.set_printoptions(suppress=True)
            print('weight array: {}'.format(self.Weight))

        for i in range(0, self.ND):
            self.Weight = 0
            if self.eps < self.Optim[i] < self.C - self.eps:
                for j in range(0, self.ND):
                    self.Weight0 += self.Y[j] * self.Optim[j] * self.KernelType(self.X[j][:], self.X[i][:])
                self.Weight0 -= self.Y[i]
                if display == True:
                    print('Weight0 = {}'.format(self.Weight0))
                break

    def calculate_pixel(self):
        VecArr = np.array([[self.px / 100], [self.py / 100]])
        Class = 0
        for i in range(0, self.ND):
            Class += self.Y[i] * self.Optim[i] * self.KernelType(self.X[i][:], VecArr)
        Class -= self.Weight0
        return Class

    def leave_one_out(self):
        LOO = np.logspace(-1, 2, num=10, endpoint=True)
        for i in range(0, LOO.size):
            self.C = LOO[i]
            Err = 0
            for j in range (0, len(self.Dots)):
                LeaveDot = self.dots_pop(0)
                self.px = LeaveDot[0]
                self.py = LeaveDot[1]
                self.calculate_weight(False)
                if np.sign(self.calculate_pixel()) != LeaveDot[3]:
                    Err += 1
                self.dots_setup(LeaveDot)
            self.LOOErr.append(Err)
        print('LOO out: {}'.format(self.LOOErr))
        self.C = LOO[self.LOOErr.index(min(self.LOOErr))]

    def Main(self):
        #build image
        self.ND = len(self.Dots)
        self.build_image()
        #optimize C parameter with LOO
        print('Starting LOO: ' + self.KernelType.__name__)
        self.leave_one_out()
        self.pix.set_title(self.KernelType.__name__ +
                           '; C={0:.2f}'.format(self.C))
        #SVM with optimize C parameter
        print('Starting SVM: ' + self.KernelType.__name__)
        self.calculate_weight()
        for self.px in range(0, 100):
            for self.py in range(0, 100):
                Class = self.calculate_pixel()
                if -1 < Class < 1:
                    self.ArrayZ[self.py, self.px] = np.sign(Class) / 2
                else:
                    self.ArrayZ[self.py, self.px] = np.sign(Class)
        #draw dot with his class
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
