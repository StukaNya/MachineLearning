from range_algorithm import *
from scipy.optimize import minimize


class SvmAlgorithm(RangeAlgorithm):

    def __init__(self, fig, NumbKer, NumbSubPlot, KernelType):
        super().__init__(fig, NumbKer, NumbSubPlot)
        # type of Vectors kernel
        self.KernelType = KernelType
        # parameter
        self.C = 100

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
        self.Y = np.zeros((self.ND, 1), dtype=float)
        # x0 for func, row
        self.L0 = np.zeros((self.ND, 1), dtype=float)
#        for i in range(0, self.ND):
#            self.L0[i] = abs(random.normalvariate(0, 0.001))

    def build_image(self):
        super().build_image()
        self.pix.set_title('SVM classification')

    def calculate_weight(self):
        self.init_arrays()
        # fill X and Y massives
        for i in range(0, self.ND):
            self.Y[i] = self.Dots[i][3]
            self.X[i, 0] = self.Dots[i][0]
            self.X[i, 1] = self.Dots[i][1]
        for i in range(0, self.ND):
            for j in range(0, self.ND):
                self.KerArr[i][j] = (self.Y[i] * self.Y[j] *
                                     self.KernelType(self.X[i][:],
                                                     self.X[j][:]))

        Lang = lambda Lamb: -1.0 * Lamb.sum() + 0.5 * np.sum(np.transpose(self.Y * Lamb).dot(self.KerArr.dot(self.Y * Lamb)))

        cons = [{'type': 'eq', 'fun': lambda Lamb: Lamb.dot(self.Y)}]
        opt = {'disp': True, 'eps': 1.4901161193847656e-08,
               'maxiter': 100, 'ftol': 1e-06}
        bound = tuple([(0, self.C)] * self.ND)

#        fx = Lang(self.L0)
#        print('x0: {} bound: {}'.format(fx, np.transpose(self.L0)))

        res = minimize(Lang, self.L0, method='SLSQP', bounds=bound,
                       jac=False, constraints=cons, options=opt)
        self.Optim = res.x
        print('optim array: {}'.format(self.Optim))

    def calculate_pixel(self):
        self.Weight0 = 0
        for i in range(0, self.ND):
            if self.Optim[i] > 0 and self.Optim[i] < self.C:
                self.Weight0 = self.KerArr[i][:].dot(self.Optim) - self.Y[i]
                break
        for self.px in range(0, 100):
            if (self.px % 10 == 0):
                print('{0} iterate'.format(self.px))
            for self.py in range(0, 100):
                VecArr = np.array([[self.px], [self.py]])
                Class = 0
                for i in range(0, self.ND):
                    Class += (self.Optim[i] * self.Y[i] *
                              self.KernelType(self.X[i][:], VecArr) -
                              self.Weight0)
                self.ArrayZ[self.px, self.py] = Class

    def Main(self):
        self.ND = len(self.Dots)
        self.build_image()
        print('Starting SVM algorithm')
        self.calculate_weight()
        self.calculate_pixel()
        for i in range(0, self.ND):
            if self.Optim[i] == 0:
                self.draw_dot(i, 40)
            else:
                self.draw_dot(i, 80)
        self.im.set_data(self.ArrayX, self.ArrayY, self.ArrayZ)
        self.pix.images.append(self.im)
        self.cb = plt.colorbar(self.im)
