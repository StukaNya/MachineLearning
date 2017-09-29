import math
import numpy as np
from random import random
from operator import add
from functools import reduce
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib.colors import LinearSegmentedColormap



def get_range(x, y):
	return (x[0] - y[0])**2 + (x[1] - y[1])**2


def get_ward_range(w, s):
	lw = 1
	ls = 1

	if type(w).__module__ != np.__name__:
		lw = len(w)
		sum_w = reduce(lambda a, x: a + x, w) / lw
	else:
		sum_w = w
	
	if type(s).__module__ != np.__name__:
		ls = len(s)
		sum_s = reduce(lambda a, x: a + x, s) / ls
	else:
		sum_s = s
	
	return get_range(sum_w, sum_s) ** 2 + ls*lw/(ls+lw)



class HierarchicalCluster(object):
	def __init__(self, fig):
		self.fig = fig
		self.clusters = []
		self.dist = np.asarray([], dtype=float)
		self.argdelta = 5
		self.delta = 0.0
		# plot arrays
		self.ArrayX = np.linspace(0, 100, 100)
		self.ArrayY = np.linspace(0, 100, 100)
		self.ArrayZ = self.ArrayX[:, np.newaxis] + self.ArrayY[np.newaxis, :]


	def get_lance_williams_range(self, idx_u, idx_v, idx_s):
		# Ward range
		ls = float(len(self.clusters[idx_s]))
		lu = float(len(self.clusters[idx_u]))
		lv = float(len(self.clusters[idx_v]))
		lw = lu + lv

		alpha_u = (ls + lu) / (ls + lw)
		alpha_v = (ls + lv) / (ls + lw)
		beta = -ls / (ls + lw)

		r_us = self.dist[idx_u][idx_s]
		r_vs = self.dist[idx_v][idx_s]
		r_uv = self.dist[idx_u][idx_v]

		return alpha_u*r_us + alpha_v*r_vs + beta*r_uv


	def build_image(self):
		self.pix = self.fig.add_subplot(111)
		self.pix.set_title('Hierarchical clustering')
		self.im = NonUniformImage(self.pix, interpolation='bilinear',
									extent=(0, 100, 0, 100), cmap='seismic')
		self.pix.set_xlim(0, 100)
		self.pix.set_ylim(0, 100)


	def draw_clusters(self):
		plt.cla()
		N = len(self.clusters)
		colors = [(1,1,1)] + [(random(),random(),random()) for i in range(0,N)]
		rand_map = LinearSegmentedColormap.from_list('new_map', colors, N=N)
		for i, cluster in enumerate(self.clusters):
			if type(cluster).__module__ != np.__name__:
				for dot in cluster:
					self.pix.scatter(dot[0], dot[1], s=40, c=colors[i+1], cmap=rand_map)
			else:
				self.pix.scatter(cluster[0], cluster[1], s=40, c=colors[i+1], cmap=rand_map)
		plt.draw()


	def init_clusters(self):
		# training data
		mu = [[25, 25], [75, 75], [25,75]]
		cov = [([80, 0], [0, 80]), ([80, 20], [20, 80]), ([80, 30], [30, 80])]
		numb = [50, 40, 38]
		N = sum(numb)
		self.dist = [[0] * N] * N

		for i in range(0,3):
			X = np.random.multivariate_normal(mu[i],cov[i],numb[i])
			for j in range(0,numb[i]):
				self.clusters.append(X[j])
		
		for i in range(0, N):
			self.dist[i] = list(map(get_range, [self.clusters[i]]*N, self.clusters))
		self.dist  = np.asarray(self.dist, dtype=float)

		argsort = np.dstack(np.unravel_index(np.argsort(self.dist.ravel()),	(N, N)))
		# set of minimal elements
		self.idx = (argsort[0,N:N+2*self.argdelta])[::2]
		#self.delta_array = self.idx[0:self.argdelta]


	def algo_iteration(self):
		N = self.dist.shape[0]
		#if len(self.clusters) < 2*self.argdelta:
		#	argsort = np.dstack(np.unravel_index(np.argsort(self.dist.ravel()),	(N, N)))
		#	self.idx = (argsort[0,N:N+len(self.clusters)])[::2]
		#	self.delta = self.dist[self.idx[-1,0], self.idx[-1,1]]
		#if self.idx.size < 3:
		argsort = np.dstack(np.unravel_index(np.argsort(self.dist.ravel()),	(N, N)))
		self.idx = (argsort[0,N:N+2*self.argdelta])[::2]
		self.delta = self.dist[self.idx[-1,0], self.idx[-1,1]]
		
		idx_set = set(tuple(x) for x in self.idx.tolist())
		idx_list = [list(x) for x in idx_set]
		self.idx = np.asarray(list(filter(lambda x: x[0] != x[1], idx_list)))
		#print('idx {}'.format(self.idx))

		uv_idx = np.sort(self.idx[0])
		self.idx = np.delete(self.idx, 0, axis=0)

		#print("------------")
		#print('delta {}'.format(self.delta))
		#print('uv_idx {}'.format(uv_idx))
		
		new_dist = np.zeros((1, self.dist.shape[0]))

		for i, cluster in enumerate(self.clusters):
			ward_range = self.get_lance_williams_range(uv_idx[0], uv_idx[1], i)
			# add new cluster to delta_idx
			#if ward_range < self.delta and i != uv_idx[0] and i != uv_idx[1]:
				#print('add {}'.format([len(self.clusters)-1, i]))
			#	self.idx = np.vstack((np.asarray([self.dist.shape[0]-1, i]), self.idx))
			# update distances
			new_dist[0, i] = ward_range
		
		u = self.clusters.pop(uv_idx[1])
		if type(u).__module__ == np.__name__:
			u = [u]
		v = self.clusters.pop(uv_idx[0])
		if type(v).__module__ == np.__name__:
			v = [v]	

		self.clusters.append(u + v)

		# delete columns/rows with idx = u/v
		self.dist = np.delete(self.dist, uv_idx[1], 0)
		self.dist = np.delete(self.dist, uv_idx[1], 1)
		self.dist = np.delete(self.dist, uv_idx[0], 0)
		self.dist = np.delete(self.dist, uv_idx[0], 1)

		new_dist = np.delete(new_dist, uv_idx[1], 1)
		new_dist = np.delete(new_dist, uv_idx[0], 1)
		
		#self.idx = list(filter(lambda x: x[0]!=uv_idx[0] and x[1]!= uv_idx[0], list(self.idx)))
		#self.idx = list(filter(lambda x: x[0]!=uv_idx[1] and x[1]!= uv_idx[1], list(self.idx)))

		self.idx = list(map(lambda x: list(map(lambda y: y-1 if y > uv_idx[0] else y, list(x))), list(self.idx)))
		self.idx = np.array(list(map(lambda x: list(map(lambda y: y-1 if y > uv_idx[1] else y, list(x))), list(self.idx))))

		ln = new_dist.shape[1]
		#new_dist = np.delete(new_dist, uv_idx).reshape(1, ln-2)
		self.dist = np.vstack((self.dist, new_dist))
		self.dist = np.hstack((self.dist, np.append(new_dist, 0.0).reshape(ln+1, 1)))


	def main(self):
		self.build_image()
		self.init_clusters()
		while len(self.clusters) > 5:
			self.algo_iteration()
			self.draw_clusters()
			#k = len(self.clusters)
			#if k < 20:
			#	fig.savefig('pic_{}.png'.format(k))				
			
if __name__ == "__main__":
	fig = plt.figure()
	Cluster = HierarchicalCluster(fig)
	Cluster.main()
	plt.show()