from top_class import *
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage


class RangeAlgorithm(Top):

    def __init__(self, fig, NumbKer, NumbSubPlot):
        super().__init__(fig)
        # Numb of signify dots
        self.NumbKer = NumbKer
        # subplot position
        self.NumbSubPlot = NumbSubPlot
        # coordinates
        self.px = 0
        self.py = 0
        # pixel positive (red) and negative (blue) weights
        self.PosWeight = 0
        self.NegWeight = 0
        # pixel normal weight
        self.Weight = 0
        # for each pixel Vector = [RangeToDot[i], DotColourStr, DotColourInt]
        self.Vectors = []
        # list, inherited from Top
        self.Dots = super().dots_get()
        # plot arrays
        self.ArrayX = np.linspace(0, 100, 100)
        self.ArrayY = np.linspace(0, 100, 100)
        self.ArrayZ = self.ArrayX[:, np.newaxis] + self.ArrayY[np.newaxis, :]

    def __del__(self):
        self.Vectors.clear()
        self.dots_clear()

    # inherit from Top
    @classmethod
    def dots_get(cls):
        return cls.ExtDots

    def dots_setup(self, dot):
        super().dots_setup(dot)
        self.ND += 1

    def dots_pop(self, i):
        self.ND -= 1
        return super().dots_pop(i)


    def dots_clear(self):
        super().dots_clear()

    def draw_dot(self, i, r):
        DotColour = 'b'
        x = self.Dots[i][0]
        y = self.Dots[i][1]
        if self.Dots[i][2] == "red":
            DotColour = 'r'
        else:
            if self.Dots[i][2] == "blue":
                DotColour = 'b'
        self.pix.scatter(x, y, s=r, c=DotColour, alpha=1)

    def build_image(self):
        self.pix = self.fig.add_subplot(self.NumbSubPlot)
        self.pix.set_title('Range classification')
        self.im = NonUniformImage(self.pix, interpolation='bilinear',
                                  extent=(0, 100, 0, 100), cmap='seismic')
        self.pix.set_xlim(0, 100)
        self.pix.set_ylim(0, 100)

    def init_vectors(self):
        for i in range(len(self.Dots)):
            Vector = [
                math.sqrt((self.Dots[i][0] - self.px) ** 2 +
                          (self.Dots[i][1] - self.py) ** 2),
                self.Dots[i][2], self.Dots[i][3]
            ]
            self.Vectors.append(Vector)
        self.Vectors.sort()

    def kernel(self, x):
        if x < 0 or x > 1 or x == 1:
            return 0
        else:
            return math.exp((-x ** 2) / (1 - x ** 2))

    def calculate_kernel(self):
        self.PosWeight = 0
        self.NegWeight = 0
        for i in range(0, self.NumbKer):
            self.w = self.kernel(self.Vectors[i][0] /
                                 self.Vectors[self.NumbKer][0])
            if self.Vectors[i][1] == "red":
                self.PosWeight += 1
            if self.Vectors[i][1] == "blue":
                self.NegWeight += -1
        #    self.Weight = ((self.PosWeight - self.NegWeight) /
        #                   (self.PosWeight + self.NegWeight))
            self.Weight = self.PosWeight + self.NegWeight


    def calculate_pixel(self):
        self.init_vectors()
        self.calculate_kernel()
        self.ArrayZ[self.py, self.px] = self.Weight
        self.Vectors.clear()

    def Main(self):
        print('Starting range algorithm')
        self.build_image()
        for self.px in range(0, 100):
            if (self.px % 10 == 0):
                print('{0} iterate'.format(self.px))
            for self.py in range(0, 100):
                self.calculate_pixel()
        for i in range(self.Dots.__len__()):
            self.draw_dot(i, 40)

        self.im.set_data(self.ArrayX, self.ArrayY, self.ArrayZ)
        self.pix.images.append(self.im)
        self.cb = plt.colorbar(self.im)
