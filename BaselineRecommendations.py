import math
import itertools
import time
from MatrixOperations import convert_coo_to_csc_and_csr
from scipy import sparse


class BaselineRecommendations:
    def __init__(self, dataset):
        # Load the sparse matrix from a file
        self.training_filepath = 'matrices/{}_training.npz'.format(dataset)
        self.testing_filepath = 'matrices/{}_test.npz'.format(dataset)
        self.training_matrix_coo = self.load_sparse_matrix(self.training_filepath)
        self.test_matrix_coo = self.load_sparse_matrix(self.testing_filepath)

        self.training_matrix_csr = None
        self.test_matrix_csr = None
        self.training_matrix_csc = None
        self.test_matrix_csc = None

        self.baseline_rating = {}
        self.movie_centered = {}
        self.movie_average = {}
        self.user_centered = {}
        self.user_average = {}
        self.global_mean = 0.0

    def load_sparse_matrix(self, file_name):
        return sparse.load_npz(file_name)

    def calculate_baseline_RMSE(self):
        summed_error = 0

        # Loop through each entry in the test dataset
        for movie, user, true_rating in itertools.izip(self.test_matrix_coo.row, self.test_matrix_coo.col,
                                                       self.test_matrix_coo.data):
            # Get the baseline rating for this movie in the test set
            movie_baseline = self.movie_centered[movie]

            # Get the baseline rating for this user in the test set
            user_baseline = self.user_centered[user]
            estimated_rating = movie_baseline + user_baseline + self.global_mean
            self.baseline_rating[(movie, user)] = estimated_rating

            # Calculate the error between the predicted rating and the true rating
            summed_error = summed_error + self.calculate_error_test(estimated_rating, true_rating)

        # Calculate the number of entries in the test set
        test_dataset_size = self.test_matrix_coo.nnz

        # Compute the RMSE on the test set
        rmse = math.sqrt(float(summed_error) / test_dataset_size)

        return rmse


    def calculate_error_test(self, estimated_rating, true_rating):
        error = math.pow(true_rating - estimated_rating, 2)
        return error


    def calculate_global_baseline_rating(self):
        summed_movie_rating = 0
        for i, j, v in itertools.izip(self.training_matrix_coo.row, self.training_matrix_coo.col,
                                      self.training_matrix_coo.data):
            summed_movie_rating = summed_movie_rating + v

        number_of_ratings = self.training_matrix_coo.nnz
        self.global_mean = float(summed_movie_rating) / number_of_ratings


    def calculate_relative_mean_movie_rating(self):
        # Calculate the mean of each movie
        movie_sums = self.training_matrix_csr.sum(axis=1)
        # Calculate the number of ratings for each movie
        movie_rating_counts = self.training_matrix_csr.getnnz(axis=1)

        # Loop through each movie
        number_of_movies = self.training_matrix_csr.shape[0]
        for index in xrange(1, number_of_movies):
            # Check to see if the movie has not been rated
            if movie_sums[index] != 0:
                movie_average = float(movie_sums[index]) / movie_rating_counts[index]
                self.movie_average[index] = movie_average
                self.movie_centered[index] = movie_average - self.global_mean
            else:
                self.movie_average[index] = self.global_mean
                self.movie_centered[index] = 0


    def calculate_mean_user_rating(self):
        # Calculate the mean of each user
        user_sums = self.training_matrix_csc.sum(axis=0)
        # Reshape the matrix to array form for proper indexing
        user_sums = user_sums.reshape((user_sums.size, 1))
        # Calculate the number of ratings for each user
        user_rating_counts = self.training_matrix_csc.getnnz(axis=0)

        # Loop through each user
        number_of_users = self.training_matrix_csc.shape[1]
        for index in xrange(1, number_of_users):
            # Check to see if the user has not rated
            if user_sums[index] != 0:
                user_average = float(user_sums[index]) / user_rating_counts[index]
                self.user_average[index] = user_average
                self.user_centered[index] = user_average - self.global_mean
            else:
                self.user_average[index] = self.global_mean
                self.user_centered[index] = 0


    def calculate_baseline_error(self):
        start = time.time()
        self.calculate_global_baseline_rating()
        end = time.time()
        print "Time to calculate global movie mean: " + str((end - start))

        start = time.time()
        self.calculate_relative_mean_movie_rating()
        end = time.time()
        print "Time to calculate mean movie ratings: " + str((end - start))

        start = time.time()
        self.calculate_mean_user_rating()

        end = time.time()
        print "Time to calculate mean user ratings: " + str((end - start))

        start = time.time()
        rmse = self.calculate_baseline_RMSE()
        end = time.time()
        print "Time to calculate RMSE: " + str((end - start))

        return rmse

    def run_baseline(self):
        self.training_matrix_csc, self.training_matrix_csr = convert_coo_to_csc_and_csr(self.training_matrix_coo)
        self.test_matrix_csc, self.test_matrix_csr = convert_coo_to_csc_and_csr(self.test_matrix_coo)
        print "Finished converting to csc and csr"
        rmse = self.calculate_baseline_error()
        print "RMSE Baseline: " + str(rmse)