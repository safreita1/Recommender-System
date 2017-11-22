
import scipy.sparse as sparse
import numpy as np
import random
original_matrix = sparse.load_npz('matrices/random_centered_training.npz')
orig_csr_matrix = original_matrix.tocsr()
sim_sparse_matrix = sparse.load_npz('matrices/random_similarity.npz')
sim_csr_matrix = sim_sparse_matrix.tocsr()

print "Test Diagonals"
for i in xrange(1,1000):
    if sim_csr_matrix[i,i] != 1:
        print "Error on diagonal [{}, {}]".format(i, i)

random.seed(42)

# Consistent random test
sim_dense_matrix = sim_csr_matrix.toarray()
for count in xrange(1,101):

    print "Random Pair Test Number: {}".format(count)
    row1 = random.randint(1,10000)
    row2 = random.randint(1,10000)
    #row1 = 10
    #row2 = count
    print "Rows: {} {}".format(row1, row2)
    val = sim_csr_matrix[row1, row2]

    print "Provide similarity: {}".format(sim_dense_matrix[row1, row2])
    #print "Provide similarity: {}".format(csr_matrix[row2, row1])
    actual_row1 = orig_csr_matrix.getrow(row1).toarray()[0]
    actual_row2 = orig_csr_matrix.getrow(row2).toarray()[0]
    dotprod = np.dot(actual_row1, actual_row2)
    mag1 = np.linalg.norm(actual_row1)
    mag2 = np.linalg.norm(actual_row2)
    if row1 == row2:
        similarity=1
    elif mag1 == 0 or mag2 == 0:
        similarity= 0
    else:
        similarity = dotprod / (mag1 * mag2)
    #mag1 = np.linalg.norm(actual_row1)
    #mag2 = np.linalg.norm(actual_row2)
    #dotprod = np.dot(actual_row1, actual_row2)
    #similarity = dotprod/(mag1*mag2)
    print "Computed similarity: {}".format(similarity)