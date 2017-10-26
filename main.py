from numpy import genfromtxt
import scipy.sparse as sparse
import numpy as np
import csv
import math
import time
import itertools
import pandas


class RecommenderSystems:
    def __init__(self):
        training_matrix = None
        test_matrix = None

    def write_matrix_to_csv(self, user_col, movie_row, rating_data, file_name):
        with open(file_name, 'wb') as file:
            writer = csv.writer(file)
            for ix in xrange(len(user_col)):
                writer.writerow((user_col[ix], movie_row[ix], rating_data[ix]))

    def calculate_baseline_RMSE(self, movie_ratings, user_ratings, global_movie_mean):
        summed_error = 0

        # Loop through each entry in the test dataset
        for movie, user, true_rating in zip(self.test_matrix.row, self.test_matrix.col, self.test_matrix.data):
            movie_baseline = 0
            user_baseline = 0

            # Check if a movie exists in the dictionary
            if movie in movie_ratings:
                # Get the baseline rating for this movie in the test set
                movie_baseline = movie_ratings[movie]

            # Check if a user exists in the dictionary
            if user in user_ratings:
                # Get the baseline rating for this user in the test set
                user_baseline = user_ratings[user]

            # Calculate the error between the predicted rating and the true rating
            summed_error = summed_error + self.calculate_error_test(movie_baseline, user_baseline, global_movie_mean, true_rating)

        # Calculate the number of entries in the test set
        test_dataset_size = self.test_matrix.nnz

        # Compute the RMSE on the test set
        rmse = math.sqrt(float(summed_error) / test_dataset_size)

        return rmse

    def calculate_error_test(self, movie_baseline, user_baseline, global_movie_mean, true_rating):
        estimated_rating = movie_baseline + user_baseline + global_movie_mean

        # Square the error
        error = math.pow(true_rating - estimated_rating, 2)
        return error

    def calculate_global_mean_movie_rating(self):
        summed_movie_rating = 0
        for i,j,v in itertools.izip(self.training_matrix.row, self.training_matrix.col, self.training_matrix.data):
            summed_movie_rating = summed_movie_rating + v

        number_of_ratings = self.training_matrix.nnz
        mean_movie_rating = float(summed_movie_rating) / number_of_ratings

        return mean_movie_rating

    def calculate_mean_movie_rating(self, global_movie_mean):
        movie_ratings = {}
        for row in self.training_matrix.row:
            row_data = self.training_matrix.getrow(row)

            sum_movie = row_data.sum()
            number_movie = row_data.nnz
            movie_average = float(sum_movie) / number_movie

            movie_ratings[row] = movie_average - global_movie_mean

        return movie_ratings

    def calculate_mean_user_rating(self, global_movie_mean):
        user_ratings = {}
        for col in self.training_matrix.col:
            col_data = self.training_matrix.getcol(col)

            sum_user = col_data.sum()
            number_users = col_data.nnz
            user_average = float(sum_user) / number_users

            user_ratings[col] = user_average - global_movie_mean

        return user_ratings

    # Remove any entries from the dataset that don't correspond to a rating
    def preprocess_data(self, file_name):
        csv_delimeter = ','

        headers = ["users", "movies", "rating"]
        rating_data = pandas.read_csv(file_name, sep=csv_delimeter, names=headers)

        # Remove any rows that contain a NaN in them
        rating_data.dropna(how='any')

        # Remove any rows that have a rating of -1
        rating_data = rating_data[rating_data.rating != -1]

        # Write the data to a csv file
        rating_data.to_csv('preprocessed_dataset.csv', header=False, index=False)


    # This should be better sampled to ensure that every movie and user is well represented in both
    # the test and training set split. In addition, the shape of the matrices needs to be properly
    # inputed.
    def create_training_test_split(self):
        # Load data in from dataset
        rating_data = genfromtxt('preprocessed_dataset.csv', delimiter=',')

        # Find the 80-20 split point from the loaded data
        split_delimiter = int(.8 * len(rating_data[:,0]))

        # 80% of the data goes to the training set
        training_user_col = rating_data[:split_delimiter,0]
        training_movie_row = rating_data[:split_delimiter,1]
        training_rating_data = rating_data[:split_delimiter,2]

        # Write the training data to a csv file
        self.write_matrix_to_csv(training_user_col, training_movie_row, training_rating_data, 'training_data.csv')

        # 20% of the data goes to the test set
        test_user_col = rating_data[split_delimiter:,0]
        test_movie_row  = rating_data[split_delimiter:,1]
        test_rating_data = rating_data[split_delimiter:,2]

        # Write the test data to a csv file
        self.write_matrix_to_csv(test_user_col, test_movie_row, test_rating_data, 'test_data.csv')

        self.training_matrix = sparse.coo_matrix((training_rating_data, (training_movie_row, training_user_col)), shape=(40000, 73516))
        self.test_matrix = sparse.coo_matrix((test_rating_data, (test_movie_row, test_user_col)), shape=(40000, 73516))

    def save_sparse_matrix(self, file_name, sparse_matrix):
        sparse.save_npz(file_name, sparse_matrix)

    def load_sparse_matrix(self, file_name):
        return sparse.load_npz(file_name)

    def initialize_program(self, load_matrix, path_to_dataset):
        # Load the sparse matrix from a file
        if load_matrix:
            self.training_matrix = self.load_sparse_matrix('matrices/training_matrix.npz')
            self.test_matrix = self.load_sparse_matrix('matrices/test_matrix.npz')
        # Process the dataset, load it into a sparse matrix, save the sparse matrix into a file
        else:
            self.preprocess_data(path_to_dataset)
            self.create_training_test_split()
            self.save_sparse_matrix('matrices/training_matrix.npz', self.training_matrix)
            self.save_sparse_matrix('matrices/test_matrix.npz', self.test_matrix)


    def calculate_baseline_error(self):
        global_movie_mean = self.calculate_global_mean_movie_rating()
        movie_ratings = self.calculate_mean_movie_rating(global_movie_mean)
        user_ratings = self.calculate_mean_user_rating(global_movie_mean)
        return self.calculate_baseline_RMSE(movie_ratings, user_ratings, global_movie_mean)


if __name__ == '__main__':
    start_whole = time.time()

    recommender = RecommenderSystems()
    recommender.initialize_program(load_matrix=True, path_to_dataset='rating_sample.csv')
    rmse = recommender.calculate_baseline_error()
    print "RMSE Baseline: " + str(rmse)



    end = time.time()
    print "Time to run program" + str((end - start_whole))


###scale to full dataset,  better training/test split

