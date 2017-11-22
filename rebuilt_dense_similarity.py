
import scipy.sparse as sparse

import time
import os
init_time = time.time()
#tick = time.time()
#data = genfromtxt('csv/similarity_matrix_arbitray.csv', delimiter=',',  dtype=float)
#print "Time elapsed: ".format(time.time()-tick)
#sparse_matrix = sparse.coo_matrix(data)
sparse_matrix = sparse.load_npz('matrices/similarity_matrix_arbitrary.npz')#, sparse_matrix)
print "Total Time elapsed: {}".format(time.time()-init_time)