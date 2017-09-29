from top_class import *
import random


class NormVariateDots(Top):

    def __init__(self, fig, NumberOfDots, CenterX,
                 CenterY, Sigma, DotColourStr):
        super().__init__(fig)
        self.NumberOfDots = NumberOfDots
        # center of normal variate
        self.CenterX = CenterX
        self.CenterY = CenterY
        # sigma of normal variate
        self.Sigma = Sigma
        # red dots have +1 sign weight, blue dots have -1, others colour have 0
        self.DotColourStr = DotColourStr

    # inherit from Top
    def dots_setup(self, dot):
        super().dots_setup(dot)

    # inherit from Top
    def dots_clear(self):
        super().dots_clear()

    def init_dots(self):
        for i in range(0, self.NumberOfDots):
            x = 101
            y = 101
            if self.DotColourStr == "red":
                self.DotColourInt = 1
            else:
                if self.DotColourStr == "blue":
                    self.DotColourInt = -1
                else:
                    self.DotColourInt = 0
            while x < 0 or x > 100:
                x = random.normalvariate(self.CenterX, self.Sigma)
            while y < 0 or y > 100:
                y = random.normalvariate(self.CenterY, self.Sigma)
            dot = [x, y, self.DotColourStr, self.DotColourInt]
            self.ExtDots.append(dot)

    def Main(self):
        self.init_dots()
        print('Init {} {} dots'.format(self.NumberOfDots, self.DotColourStr))
