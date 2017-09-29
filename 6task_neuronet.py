import math
import random
import importimg
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


# derivative of sigmoid
def dsigmoid(y):
	return y * (1.0 - y)


class NeuralNetwork(object):
	def __init__(self, train_data, test_data, hidden_n):
		# load input images and output labels
		self.train_images = train_data[0]
		self.train_labels = train_data[1]
		self.test_images = test_data[0]
		self.test_labels = test_data[1]
		# add 1 for bias node
		self.input = self.train_images.shape[1] + 1
		self.hidden = hidden_n
		self.output = 10
		# set up array of 1s for activations
		self.ai = np.ones((self.input,), dtype=np.float)
		self.ah = np.ones((self.hidden,), dtype=np.float)
		self.ao = np.ones((self.output,), dtype=np.float)
		# create randomized weights
		self.wi = np.random.randn(self.input, self.hidden) 
		self.wo = np.random.randn(self.hidden, self.output) 
		
		if __name__ == "__main__":
			print('Create Neural Network with size: ' +
			'(input, hidden, output) = ' + 
			'{}'.format((self.input, self.hidden, self.output)))


	def feedForward(self, layers_i):
		if len(layers_i) != self.input - 1:
			raise ValueError('Wrong number of inputs: {}'.format(len(inputs)))

		# input activations
		self.ai[0:self.input-1] = layers_i[0:self.input-1]
		self.ai[self.input-1] = -1
		# hidden activations
		for i in range(0, self.hidden):
			self.ah[i] = sigmoid(self.ai.dot(self.wi[:,i]))
		# output activations
		for i in range(0, self.output):
			self.ao[i] = sigmoid(self.ah.dot(self.wo[:,i]))
		
		return self.ao[:]


	def backPropagate(self, targets, N):
		if len(targets) != self.output:
			raise ValueError('Wrong number of targets!')
		# calculate error terms for output
		# the delta tell you which direction to change the weights
		output_deltas = np.zeros((self.output,), dtype=np.float)
		for i in range (0,self.output):
			output_deltas[i] = self.ao[i] - targets[i]
		# calculate error terms for hidden
		# delta tells you which direction to change the weights
		hidden_deltas = np.zeros((self.hidden,), dtype=np.float)
		for i in range(0, self.hidden):
			hidden_deltas[i] = dsigmoid(self.ah[i])*(output_deltas.dot(self.wo[i,:]))
		# update the weights connecting hidden to output
		for j in range(0, self.hidden):
			for k in range(0, self.output):
				change = output_deltas[k] * self.ah[j] * dsigmoid(self.ao[k])
				self.wo[j][k] -= N * change

		# update the weights connecting input to hidden
		for i in range(self.input):
			for j in range(self.hidden):
				change = hidden_deltas[j] * self.ai[i] * dsigmoid(self.ah[j])
				self.wi[i][j] -= N * change
		# calculate error
		error = 0.0
		for k in range(0, len(targets)):
			error += 0.5 * (targets[k] - self.ao[k]) ** 2
		return error


	def trainNetwork(self, rule_func, iterations=500, N=0.5):
		# N: learning rate
		if __name__ == "__main__":
			print('Start training with: ' + 
				rule_func.__name__)
		err_cnt = 0
		targets = np.zeros((self.output,1), dtype=float)
		for i in range(1, iterations):
			error = 0.0
			# 1st heuristic - random input data
			idx = np.random.randint(0, self.train_labels.shape[0], size=None)
			# 2nd heuristic - normalization pixel data (0.0-1.0)
			inputs = self.train_images[idx].astype(float) / 255.0
			# set right output values and run train
			targets[self.train_labels[idx]] = 1.0
			a_o = self.feedForward(inputs)
			error = self.backPropagate(targets, N)
			targets[self.train_labels[idx]] = 0.0
			# info output & save training set (for boosting)
			algo_resp = rule_func(a_o)
			if __name__ == "__main__":
				if i % 100 == 0:
					print('i:{};   y = {}, a = {}, eps = {}, err = {}%'
						.format(i, self.train_labels[idx], algo_resp, error, err_cnt))
					err_cnt = 0
				if self.train_labels[idx] != algo_resp:
					err_cnt += 1


	def eraseWeights(self):
		self.wi = np.random.randn(self.input, self.hidden) 
		self.wo = np.random.randn(self.hidden, self.output) 


	def testNetwork(self, rule_func, test_set_i=None, iterations=1000):
		err_cnt = 0
		if test_set_i is None:
			test_set = np.random.randint(0, self.test_labels.shape[0], size=iterations)
			iter_sz = iterations
		else:
			test_set = np.asarray(test_set_i, dtype=int)
			iter_sz = test_set.shape[0]
		
		if __name__ == "__main__":
			print('Start testing ({} iterations)'.format(iter_sz))

		test_set_o = np.zeros((3, iter_sz), dtype=int)
		targets = np.zeros((self.output,1), dtype=float)
		for i in range(0, iter_sz):
			#idx = np.random.randint(0, self.labels.shape[0], size=None)
			idx = test_set[i]
			inputs = self.test_images[idx].astype(float) / 255.0
			a_o = self.feedForward(inputs)
			if rule_func(a_o) != self.test_labels[idx]:
				err_cnt += 1
			test_set_o[:,i] = np.asarray([idx, self.test_labels[idx], rule_func(a_o)])

		if __name__ == "__main__":
			err_perc = float(err_cnt)/float(iterations)*100.0
			print('Test done! Errors  {:.2f}%'.format(err_perc))
			
		return test_set_o


	def readWeights(self):
		try:
			self.wi = np.genfromtxt('wi.csv', delimiter=",")
		except IOError:
			self.wi = np.array([])
			print('Load error! (in weights)')
		try:
			self.wo = np.genfromtxt('wo.csv', delimiter=",")
		except IOError:
			self.wo = np.array([])
			print('Load error! (out weights)')


	def writeWeights(self):
		try:
			np.savetxt('wi.csv', self.wi, delimiter=',')
		except IOError:
			print('IOerror! (wi)') 
		else:   
			print('Save in weights')
		try:
			np.savetxt('wo.csv', self.wo, delimiter=',')
		except IOError:
			print('IOerror! (wo)') 
		else:   
			print('Save out weights')


# run script
if __name__ == "__main__":	 
	train_data = importimg.LoadMNIST("train-images-idx3-ubyte.gz", 
									"train-labels-idx1-ubyte.gz")
	test_data = importimg.LoadMNIST("t10k-images-idx3-ubyte.gz", 
									"t10k-labels-idx1-ubyte.gz")

	hidden_n = 50

	rule_func = lambda x: np.argmax(x)
	rule_func.__name__ = 'Argmax classifier'

	NeuNet = NeuralNetwork(train_data, test_data, hidden_n)

	NeuNet.trainNetwork(rule_func)
	NeuNet.writeWeights()

	NeuNet.readWeights()
	NeuNet.testNetwork(rule_func)



