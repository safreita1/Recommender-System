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



