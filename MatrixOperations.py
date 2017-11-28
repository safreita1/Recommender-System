from scipy import sparse
import itertools
import numpy as np

def center_matrix(sparse_matrix, file_name, movie_average):
    ix = 0
    num_movies = sparse_matrix.shape[0]
    num_users = sparse_matrix.shape[1]
    num_ratings = len(sparse_matrix.data)
    ratings = np.zeros((num_ratings))
    movies = np.zeros((num_ratings))
    users = np.zeros((num_ratings))

    # Create vectors of centered ratings
    for movie, user, rating in itertools.izip(sparse_matrix.row, sparse_matrix.col, sparse_matrix.data):
        movies[ix] = movie
        users[ix] = user
        ratings[ix] = rating - movie_average[movie]
        ix = ix + 1

    new_sparse_matrix = sparse.coo_matrix((ratings, (movies, users)), shape=(num_movies, num_users), dtype=np.float64)
    sparse.save_npz(file_name, new_sparse_matrix)


def center_matrix_user(sparse_matrix, user_average):
    ix = 0
    num_movies = sparse_matrix.shape[0]
    num_users = sparse_matrix.shape[1]
    num_ratings = len(sparse_matrix.data)
    ratings = np.zeros((num_ratings))
    movies = np.zeros((num_ratings))
    users = np.zeros((num_ratings))

    # Create vectors of centered ratings
    for movie, user, rating in itertools.izip(sparse_matrix.row, sparse_matrix.col, sparse_matrix.data):
        movies[ix] = movie
        users[ix] = user
        ratings[ix] = rating - user_average[user]
        ix = ix + 1

    centered_matrix = sparse.coo_matrix((ratings, (movies, users)), shape=(num_movies, num_users), dtype=np.float64)

    return centered_matrix

def convert_coo_to_csr(matrix):
    return matrix.tocsr()

def convert_coo_to_csc(matrix):
    return matrix.tocsc()

def convert_coo_to_csc_and_csr(matrix):
    return matrix.tocsc(), matrix.tocsr()


def test_center(self):
    new_sparse_matrix = sparse.load_npz('matrices/centered_training_arbitrary.npz')
    sparse_matrix = self.training_matrix_coo.tocsr()
    new_sparse_matrix = new_sparse_matrix.tocsr()
    ix = 0
    flag = True

    # Iterate over all nonempty rows in csr
    while ix < sparse_matrix.shape[0]:
        num = 0
        while num < 1 and ix < sparse_matrix.shape[0]:
            new_row = new_sparse_matrix.getrow(ix)
            old_row = sparse_matrix.getrow(ix)
            num = new_row.nnz
            ix = ix + 1
            if num != old_row.nnz:
                print "why"
        # Verify the new rating = old rating - mean
        for index in new_row.indices:

            if index in old_row.indices:
                old_data = sparse_matrix[ix, index]
                mean = self.movie_average[ix]
                new_data = new_sparse_matrix[ix, index]
                if old_data - mean != new_data and old_data != 0 and new_data != 0:
                    flag = False
                    print "Value Mismatch: Expected {} Got {} For {} With {}".format(old_data + mean,new_data, ix, mean)

            else:
                print "index mismatch"

        ix = ix + 1
    print "Clean? {}".format(flag)



