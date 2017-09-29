import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib import cm

def kernel(r, kern):
    if (r > 1):
        return 0

    if(kern == 'E'):
        return (3*(1-r**2))/4
    if(kern == 'T'):
        return (1-r)
    if(kern == 'Q'):
        return (15*((1-r**2)**2))/16
    else:
        print ("No desired kernel found")
        raise SystemExit

def parsen_predict(X_train, y_train, X_test, h, kern):

    print ("Starting Parsen window prediction...")

    y_predict = np.zeros(len(X_test))

    for i in range(len(X_test)):
        p1 = 0
        p2 = 0
        for j in range(len(X_train)):
            d = np.linalg.norm(X_train[j] - X_test[i])
            if(y_train[j]==1):
                p1 += kernel(d/float(h), kern)
                p2 += 0
            if(y_train[j]==-1):
                p1 += 0
                p2 += kernel(d/float(h), kern)
        if(p1 > p2):
            y_predict[i] = 1*p1/float(len(X_train))
        else:
            y_predict[i] = -1*p2/float(len(X_train))

    print ("Done.")
    return y_predict

def parameter_predict(X_train, y_train, X_test, ):

    print ("Starting parameter Bayes prediction...")

    y_predict = np.zeros(len(X_test))

    n1 = 0
    n2 = 0
    mu_1 = np.array([0,0])
    mu_2 = np.array([0,0])
    S_1 = np.zeros((2,2))
    S_2 = np.zeros((2,2))
    for i in range(len(y_train)):
        if(y_train[i] == 1):
            n1 = n1 + 1
            mu_1 = mu_1 + X_train[i]
        else:
            n2 = n2 + 1
            mu_2 = mu_2 + X_train[i]

    mu_1 = mu_1/float(n1)
    mu_2 = mu_2/float(n2)

    for i in range(len(y_train)):
        if(y_train[i] == 1):
            S_1 = np.add(S_1, np.dot(np.matrix(X_train[i] - mu_1).T, np.matrix(X_train[i] - mu_1)))
        else:
            S_2 = np.add(S_2, np.dot(np.matrix(X_train[i] - mu_2).T, np.matrix(X_train[i] - mu_2)))

    S_1 = S_1 / float(n1)
    S_2 = S_2 / float(n2)

    S_1_inv = np.linalg.inv(S_1)
    S_2_inv = np.linalg.inv(S_2)

    print ("Class 1 calculated distribution mean =", mu_1)
    print ("Class 1 calculated covariance matrix =", S_1.reshape(1, 4))
    print ("Class 2 calculated distribution mean =", mu_2)
    print ("Class 2 calculated covariance matrix =", S_2.reshape(1, 4))

    P_1 = n1/float(len(y_train))
    P_2 = n2/float(len(y_train))

    for i in range(len(X_test)):
        p1 = np.log(P_1) - 0.5*np.dot(np.dot(np.matrix(X_test[i]-mu_1), S_1_inv), np.matrix(X_test[i]-mu_1).T) - 0.5*np.log(np.linalg.det(S_1))
        p2 = np.log(P_2) - 0.5*np.dot(np.dot(np.matrix(X_test[i]-mu_2), S_2_inv), np.matrix(X_test[i]-mu_2).T) - 0.5*np.log(np.linalg.det(S_2))
        if(p1 > p2):
            y_predict[i] = 1
        else:
            y_predict[i] = -1

    print ("Done.")
    return y_predict


def create_normal_data():

    #training data
    m1 = [40, 40]
    c1 = ([80, 5], [5, 80])
    m2 = [60, 60]
    c2 = ([80, 1], [1, 80])
    X1 = np.random.multivariate_normal(m1, c1, 70)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(m2, c2, 70)
    y2 = np.ones(len(X2)) * -1
    print ("Class 1 real distribution mean =", m1)
    print ("Class 1 real covariance matrix =", c1)
    print ("Class 2 real distribution mean =", m2)
    print ("Class 2 real covariance matrix =", c2)
    return X1, y1, X2, y2

def Parsen_demo():

    #fix seed of np randon function
    np.random.seed(1999)

    X1_train, y1_train, X2_train, y2_train = create_normal_data()

    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))

    kern = 'E'

    h = 10

    # making test data (X_test) in [0, 1]
    xmin = 0
    xmax = 100
    ymin = 0
    ymax = 100
    nx = 100
    ny = 100

    x1, x2 = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    x1, x2 = np.meshgrid(x1, x2)
    x1 = x1.reshape((nx*ny, 1))
    x2 = x2.reshape((nx*ny, 1))

    X_test = np.concatenate((x1, x2), axis = 1)

    # prediction
    y_test = parsen_predict(X_train, y_train, X_test, h, kern)

    # plotting results of prediction
    fig, ax = plt.subplots()
    plt.axis([xmin, xmax, ymin, ymax])
    print (y_test)
    img_data = y_test.reshape((nx, ny))

    colorbar_range = np.amax(np.absolute(y_test))

    cax = ax.imshow(img_data, interpolation='nearest', cmap=cm.bwr, vmin = -colorbar_range, vmax = colorbar_range)
    ax.set_title('Bayes')

    #draw vertical colorbar
    cbar = fig.colorbar(cax, ticks=[-5, 0, 5])

    #plot train data
    plt.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    plt.plot(X2_train[:, 0], X2_train[:, 1], "bo")

    #plt.axis("tight")
    plt.show()


def LOO_demo():

    #fix seed of np randon function
    np.random.seed(1999)

    # making train data (X_train) and labels (y_train)
    X1_train, y1_train, X2_train, y2_train = create_normal_data()

    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))

    kern = 'E'
    n = 100
    HH = np.linspace(2, 150, n)
    Q = np.zeros((len(HH), 2))
    for i in range(len(HH)):
        Q[i, 0] = HH[i]
        for j in range(len(X_train)):
            X = np.delete(X_train, j, 0)
            Y = np.delete(y_train, j)
            x = np.reshape(X_train[j],(1,2))
            pred = parsen_predict(X, Y, x, Q[i, 0], kern)
            if(y_train[j]*np.sign(pred) > 0):
                Q[i,1] += 0
            else:
                Q[i,1] += 1


    plt.plot(Q[:, 0], Q[:, 1])
    plt.ylabel("Q")
    plt.xlabel("h")

    plt.show()

    h = HH[np.argmin(Q[:, 1])]
    Q_min = np.min(Q[:, 1])
    print ("min h = ", h)
    print ("min Q = ", Q_min)

    # making test data (X_test) in [0, 1]
    xmin = 0
    xmax = 100
    ymin = 0
    ymax = 100
    nx = 100
    ny = 100

    x1, x2 = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    x1, x2 = np.meshgrid(x1, x2)
    x1 = x1.reshape((nx*ny, 1))
    x2 = x2.reshape((nx*ny, 1))

    X_test = np.concatenate((x1, x2), axis = 1)

    # prediction
    y_test = parsen_predict(X_train, y_train, X_test, h, kern)

    # plotting results of prediction
    fig, ax = plt.subplots()
    plt.axis([xmin, xmax, ymin, ymax])
    img_data = y_test.reshape((nx, ny))

    colorbar_range = np.amax(np.absolute(y_test))

    cax = ax.imshow(img_data, interpolation='nearest', cmap=cm.bwr, vmin = -colorbar_range, vmax = colorbar_range)
    ax.set_title('Bayes')

    #draw vertical colorbar
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])

    #plot train data
    plt.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    plt.plot(X2_train[:, 0], X2_train[:, 1], "bo")

    #plt.axis("tight")
    plt.show()

def Parameter_demo():

    #fix seed of np randon function
    np.random.seed(1999)

    X1_train, y1_train, X2_train, y2_train = create_normal_data()

    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))

    # making test data (X_test) in [0, 1]
    xmin = 0
    xmax = 100
    ymin = 0
    ymax = 100
    nx = 100
    ny = 100

    x1, x2 = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    x1, x2 = np.meshgrid(x1, x2)
    x1 = x1.reshape((nx*ny, 1))
    x2 = x2.reshape((nx*ny, 1))

    X_test = np.concatenate((x1, x2), axis = 1)

    # prediction
    y_test = parameter_predict(X_train, y_train, X_test)

    # plotting results of prediction
    fig, ax = plt.subplots()
    plt.axis([xmin, xmax, ymin, ymax])
    img_data = y_test.reshape((nx, ny))

    def Q_test():
        Q = 0
        i = 0
        while i < len(X_train):
            if parameter_predict(X_train, y_train, [X_train[i]]) != y_train[i]:
                Q += 1
            else:
                Q += 0
            i += 1

        print ("Q = ", Q)

    #Q_test()

    cax = ax.imshow(img_data, interpolation='nearest', cmap=cm.coolwarm)
    ax.set_title('Bayes')

    #draw vertical colorbar
    cbar = fig.colorbar(cax, ticks=[-5, 0, 5])

    #plot train data
    plt.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    plt.plot(X2_train[:, 0], X2_train[:, 1], "bo")

    #plt.axis("tight")
    plt.show()


Parsen_demo()
#LOO_demo()
#Parameter_demo()

