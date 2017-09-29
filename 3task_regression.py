from matplotlib import pyplot as plt
import numpy as np

class OLS:

	def __init__(self, percent, range_, csv):
		self.N = 10 ** 4
		# x = 40
		self.x = int(csv.shape[1]) - 1
		# y = 80
		self.y = int(csv.shape[0])
		self.range_ = range_
		# matrix csv = (80,41) = (row, column)
		# Y (80,1)
		self.Y = np.real(csv[0:self.y, self.x])
		# F (80,40)
		self.F = np.real(csv[0:self.y, 0:self.x])
		self.per = int(np.around(self.y * percent / 100)) 
		self.iter_ = np.random.choice(self.y, size=self.per, replace = False)

	def init_data(self):
		# F and Y for learn and control data
		self.FLearn = np.zeros((self.per, self.x), dtype = float)
		self.FContr = np.zeros((self.y - self.per, self.x), dtype = float)
		self.YLearn = np.zeros((self.per,1), dtype = float)
		self.YContr = np.zeros((self.y - self.per,1), dtype = float)

		i = 0
		k = 0
		for j in range(0, self.y):
			if np.argwhere(self.iter_ == j).size != 0:
				self.FLearn[i,:] = self.F[j,:]
				self.YLearn[i] = self.Y[j]
				i += 1
			else:
				self.FContr[k,:] = self.F[j,:]
				self.YContr[k] = self.Y[j]
				k += 1

		# F = U*D*V (in numpy)		
		Utemp, self.lamb, Vtemp = np.linalg.svd(self.FLearn, full_matrices=False)

		print(self.lamb)

		# F = V*D*Ut (in lecture)
		self.U = np.transpose(Vtemp)
		self.V = Utemp

		self.lambU = np.square(self.lamb)

		self.Q = np.zeros((self.N,1), dtype=float)
		self.tauArr = np.linspace(0.0, self.range_, num = self.N)

	#	print('FLearn = {}; YLearn = {}; U = {}; V = {}'.format(self.FLearn.shape, self.YLearn.shape, self.U.shape, self.V.shape))
	#	print('norm U = {}'.format(np.linalg.norm(np.diagflat(np.ones((self.x,1), dtype=float)) - np.transpose(self.U).dot(self.U))))
	#	print('norm V = {}'.format(np.linalg.norm(np.diagflat(np.ones((self.x,1), dtype=float)) - np.transpose(self.V).dot(self.V))))
	#	print('norm SVD = {}'.format(np.linalg.norm(self.FLearn - self.V.dot(np.diag(self.lamb).dot(np.transpose(self.U))))))


	def QContr(self, tau):
		diag = np.diagflat(np.divide(self.lamb, (np.real(self.lambU) + tau))) 
		return np.linalg.norm(self.FContr.dot(self.U).dot(diag).dot(np.transpose(self.V)).dot(self.YLearn) - self.YContr)

	def QLearn(self, tau):
		diag = np.diagflat(np.divide(self.lamb, (np.real(self.lambU) + tau))) 
		return np.linalg.norm(self.FLearn.dot(self.U).dot(diag).dot(np.transpose(self.V)).dot(self.YLearn) - self.YLearn)

	def Koef(self, tau):
		diag = np.diagflat(np.divide(self.lamb, (np.real(self.lambU) + tau))) 
	#	return self.lamb[i] / (np.real(self.lambU[i]) + tau) * np.sum(self.V[:,i] * self.YLearn) * np.transpose(self.U[:,i])
		return self.U.dot(diag.dot(np.transpose(self.V).dot(self.YLearn)))

	def Main(self):
		self.init_data()
#		Koef_ = np.zeros((self.x,1))
		Koef_test = self.Koef(0.0)
		for i in range(0, self.N):
			self.Q[i] = self.QContr(self.tauArr[i])
		tauMin = self.tauArr[np.argmin(self.Q)]
		QContrReg = self.QContr(tauMin)
		QContr0 = self.QContr(1.0)
		QLearn0 = self.QLearn(1.0)
		QLearnReg = self.QLearn(tauMin)
		print('QLearn0 = {}; QLearnReg = {}'.format(QLearn0, QLearnReg))
		print('QContr0 = {}; QContrReg = {}'.format(QContr0, QContrReg))

		Koef_optim = self.Koef(tauMin)
		print('y_test //////// y_optim //////// y_data')
		for i in range(0, int(self.y-self.per)):
			print('{}, {}, {}'.format((self.FContr.dot(Koef_test))[i], (self.FContr.dot(Koef_optim))[i], self.YContr[i]))
		return self.Q


if __name__ == "__main__":
	csv = np.genfromtxt('data.csv', delimiter=",")
	range_ = 1.0

	top = OLS(75, range_, csv)
	Q = top.Main()
	tauArr = np.linspace(0.0, range_, num = 10 ** 4)

	plt.plot(tauArr, Q)
	plt.xlabel('Tau')
	plt.ylabel('Func Q')
	plt.title('OLS')
	plt.show()
