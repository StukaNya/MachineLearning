import math
import random
import neuronet
import importimg
import numpy as np
import matplotlib.pyplot as plt
from neuronet import NeuralNetwork



class AdaBoosting(object):
	def __init__(self, train_data, test_data, train_th,
				rule_func, numb_sets, hidden_n, train_n, test_sz):
		# load input images and output labels
		self.train_data = train_data
		self.train_images = train_data[0]
		self.train_labels = train_data[1]

		self.test_data = test_data
		self.test_images = test_data[0]
		self.test_labels = test_data[1]

		self.numb_sets = numb_sets
		self.train_th = train_th
		# params length check
		if not (len(hidden_n) == len(train_n) == numb_sets):
			print('Wrong parameters! Set it randomly')
			self.train_n = [np.random.randint(100, 500) for i in range(0, numb_sets)]
			self.hidden_n = [np.random.randint(10, 50) for i in range(0, numb_sets)]
		else:
			self.hidden_n = hidden_n
			self.train_n = train_n

		# rule function length check
		if callable(rule_func):
			self.rule_func = rule_func
		else:
			print('Wrong rule function! Set np.argmax(x)')
			self.rule_func = lambda x: np.argmax(x)

		self.test_sz = test_sz
		self.test_set = np.random.randint(0, len(self.test_data[1]), size=test_sz)
		self.alpha = []
		self.networks = []


	def initNeuroNetwork(self, rule_func, hidden_nu, train_nu):
		print('Init Neural Network with size of hidden - {}; '.format(hidden_nu) +
			'size of train set - {}; rule func - '.format(train_nu)
			+ rule_func.__name__)

		TempNet = NeuralNetwork(self.train_data, self.test_data, hidden_nu)
		cnt_err = self.test_sz
		test_set = []
		errors = []

		#while cnt_err > self.test_sz * self.train_th:
		TempNet.trainNetwork(rule_func,	iterations=train_nu, N=0.5)
		test_return = TempNet.testNetwork(rule_func, test_set_i=self.test_set)
		errors = np.not_equal(test_return[1,:], test_return[2,:])
		cnt_err = errors.astype(int).sum()

		print('Errors on test set {:.2f}%'.format(float(cnt_err)/float(self.test_sz)*100.0))
		self.networks.append(TempNet)

		return errors


	def setRule(self, rule_func, test_nu, errors):
		cnt_err = errors.astype(int).sum()

		e = (errors.astype(int) * self.weights).sum(dtype=float)
		alpha = 0.5 * math.log((1-e)/e)
		print('e = {:.3f}%, alpha = {:.3f}'.format(e, alpha))

		w = np.zeros((test_nu,), dtype=float)
		for i in range(0, test_nu):
			if errors[i]: 
				w[i] = self.weights[i] * math.exp(alpha)
			else: 
				w[i] = self.weights[i] * math.exp(-alpha)

		self.weights = w / w.sum()
		self.alpha.append(alpha)
	

	def trainAda(self):
		self.weights = np.ones((self.test_sz,), dtype=float) / self.test_sz
		for i in range(0, self.numb_sets):
			print('//{} iteration//'.format(i+1))
			errors = self.initNeuroNetwork(self.rule_func,
						self.hidden_n[i], self.train_n[i])
			self.setRule(self.rule_func, self.test_sz, errors)


	def testAda(self):
		print('Start testing ({} iterations)'.format(self.test_sz))
		cnt_err = 0
		for i in range(0,self.test_sz):
			a_o = np.zeros((10,), dtype=np.float)
			for j, net in enumerate(self.networks):
				temp_o = net.feedForward(self.test_images[self.test_set[i]])
				a_o += temp_o * self.alpha[j]
			response = self.rule_func(np.asarray(a_o))
			if response != self.test_labels[self.test_set[i]]:
				cnt_err += 1
		errors = float(cnt_err) / float(self.test_sz) * 100.0
		print('AdaBoost test errors {:.2f}%'.format(errors))



# run stript
if __name__ == "__main__":
	train_data = importimg.LoadMNIST("train-images-idx3-ubyte.gz",
									"train-labels-idx1-ubyte.gz")
	test_data = importimg.LoadMNIST("t10k-images-idx3-ubyte.gz",
									"t10k-labels-idx1-ubyte.gz")
	# number of networks
	numb_sets = 5
	# train threshold
	train_th = 0.2
	# each network has its own params
	hidden_n = [np.random.randint(80, 100) for i in range(0, numb_sets)]
	train_n = [np.random.randint(500, 1000) for i in range(0, numb_sets)]
	test_sz = 1000

	rule_func = lambda x: np.argmax(x)
	rule_func.__name__ = 'Argmax classifier'

	AdaBoost = AdaBoosting(train_data, test_data, train_th, rule_func, 
							numb_sets, hidden_n, train_n, test_sz)

	AdaBoost.trainAda()
	AdaBoost.testAda()