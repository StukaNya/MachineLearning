class MetaClass(type):
    def __init__(cls, name, bases, attrs):
        type.__init__(cls, name, bases, attrs)


class Top(object, metaclass=MetaClass):

    # dots is shared array of all colour dots
    # each dot = [Xcoord, Ycoord, colour]
    ExtDots = []

    def __init__(self, fig):
        # initial figure
        self.fig = fig
        self.fig.suptitle('Machine Learning')

    def dots_setup(self, dot):
        self.ExtDots.append(dot)

    def dots_pop(self, i):
        return self.ExtDots.pop(i)

    def dots_clear(self):
        self.ExtDots.clear()

    def dots_get(self):
        return self.ExtDots
