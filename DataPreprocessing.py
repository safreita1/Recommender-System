from numpy import genfromtxt
import scipy.sparse as sparse
import numpy as np
import csv
import pandas
from collections import defaultdict
import random
import time
import os

class DataPreprocessing:
    def __init__(self):
        start_time = time.time()
        print "Initializing Data Preprocessing"
        self.arbitrary_training_filepath = 'csv/arbitrary_training.csv'
        self.arbitrary_testing_filepath = 'csv/arbitrary_test.csv'
        self.random_training_filepath = 'csv/random_training.csv'
        self.random_testing_filepath = 'csv/random_test.csv'
        self.remapped_filepath = 'csv/remapped_rating.csv'
        self.movie_filepath = 'csv/anime.csv'
        self.rating_filepath = 'csv/rating.csv'
        self.npz_path = 'matrices/'

        if not os.path.isdir('csv/'):
            print "Ensure that there exists the csv folder in current working directory"
            exit(1)
        if not os.path.isfile(self.movie_filepath):
            print "Ensure anime.csv is in the csv/ folder in current working directory"
            exit(1)
        if not os.path.isfile(self.rating_filepath):
            print "Ensure rating.csv is in the csv/ folder in current working directory"
            exit(1)
        if not os.path.isdir(self.npz_path):
            os.mkdir(self.npz_path)
        if not os.path.isfile(self.remapped_filepath):
            tick = time.time()
            print "Remapped data not found\nRemapping data"
            #TODO add expected runtime
            self.remap_dataset(path_to_movies=self.movie_filepath, path_to_ratings=self.rating_filepath,
                                     target_filepath=self.remapped_filepath)
            print "Data remapped in {} seconds".format(time.time()-tick)

        if os.path.isfile(self.remapped_filepath):
            tick = time.time()
            print "Loading remapped data"
            self.rating_data = self.load_csv_as_nparray(self.remapped_filepath)
            print "Remapped data loaded in {} seconds".format(time.time()-tick)
        else:
            print "Data not found\nTerminating"
            exit(1)
        print "Data Preprocessing initialization done in {} seconds".format(time.time()-start_time)

    def save_sparse_matrix(self, file_name, sparse_matrix):
        sparse.save_npz(file_name, sparse_matrix)
    def write_matrix_to_csv(self, user_col, movie_row, rating_data, file_name):
        with open(file_name, 'wb') as file:
            writer = csv.writer(file)
            for ix in xrange(len(user_col)):
                writer.writerow((user_col[ix], movie_row[ix], rating_data[ix]))
    # Input:
    # filepath: path to data to be sampled in .csv format
    def load_csv_as_nparray(self, filepath):
        data = genfromtxt(filepath, delimiter=',', skip_header=1, dtype=float)
        return data

    def random_sample(self, data, seed=42, percent_test = 0.2):
        random.seed(seed)
        total_size = data.shape[0]
        test_size = int(total_size * (percent_test ))
        test_list = random.sample(xrange(total_size),test_size)
        test_data = np.zeros((test_size, 3))
        test_index = 0

        for ix in test_list:
            test_data[test_index]=  data[ix]
            test_index = test_index + 1
        training_data = np.delete(data, test_list, 0)

        return training_data, test_data,

    def random_split(self, rating_data, training_file_name, test_file_name):
        training_dst = self.npz_path + 'random_training'
        testing_dst = self.npz_path + 'random_test'

        # Randomly split the data into an 80-20 training/test set respectively
        training_data, test_data = self.random_sample(rating_data)

        # Training data
        training_user_col = training_data[:, 0]
        training_movie_row = training_data[:, 1]
        training_rating_data = training_data[:, 2]

        # Write the training data to a csv file
        #print "Writing training data to .csv"
        #self.write_matrix_to_csv(training_user_col, training_movie_row, training_rating_data, training_file_name)
        #print "Done"

        # Test data
        test_user_col = test_data[:, 0]
        test_movie_row = test_data[:, 1]
        test_rating_data = test_data[:, 2]

        # Write the test data to a csv file
        #print "Writing test data to .csv"
        #self.write_matrix_to_csv(test_user_col, test_movie_row, test_rating_data, test_file_name)
        #print "Done"

        # Find the sparse matrix dimensions
        sparse_user_size = max(rating_data[:, 0]) + 1
        sparse_movie_size = max(rating_data[:, 1]) + 1

        print "Building Sparse Matrices"
        training_matrix = sparse.coo_matrix((training_rating_data, (training_movie_row, training_user_col)),
                                            shape=(sparse_movie_size, sparse_user_size), dtype=np.float64)
        test_matrix = sparse.coo_matrix((test_rating_data, (test_movie_row, test_user_col)),
                                             shape=(sparse_movie_size,sparse_user_size),dtype=np.float64)

        print "Saving Sparse Matrices"
        self.save_sparse_matrix(file_name=training_dst, sparse_matrix=training_matrix)
        self.save_sparse_matrix(file_name=testing_dst, sparse_matrix=test_matrix)


    def arbitrary_split(self, rating_data, training_file_name, test_file_name):
        training_dst = self.npz_path + 'arbitrary_training'
        testing_dst = self.npz_path + 'arbitrary_test'
        # Find the 80-20 split point from the loaded data
        split_delimiter = int(.8 * len(rating_data[:, 0]))
        # 80% of the data goes to the training set
        training_user_col = rating_data[:split_delimiter, 0]
        training_movie_row = rating_data[:split_delimiter, 1]
        training_rating_data = rating_data[:split_delimiter, 2]

        # Write the training data to a csv file
        # print "Writing training data to .csv"
        # self.write_matrix_to_csv(training_user_col, training_movie_row, training_rating_data, training_file_name)
        # print "Done"

        # 20% of the data goes to the test set
        test_user_col = rating_data[split_delimiter:, 0]
        test_movie_row = rating_data[split_delimiter:, 1]
        test_rating_data = rating_data[split_delimiter:, 2]

        # Write the test data to a csv file
        # print "Writing test data to .csv"
        # self.write_matrix_to_csv(test_user_col, test_movie_row, test_rating_data, test_file_name)
        # print "Done"

        # Find the sparse matrix dimensions
        sparse_user_size = max(rating_data[:, 0])+1
        sparse_movie_size = max(rating_data[:, 1])+1

        # Create the sparse matrices for the training and test data
        print "Building Sparse Matrices"
        training_matrix = sparse.coo_matrix((training_rating_data, (training_movie_row, training_user_col)),
                                                 shape=(sparse_movie_size, sparse_user_size), dtype=np.float64)
        test_matrix = sparse.coo_matrix((test_rating_data, (test_movie_row, test_user_col)),
                                             shape=(sparse_movie_size,sparse_user_size),dtype=np.float64)

        print "Saving Sparse Matrices"
        self.save_sparse_matrix(file_name=training_dst, sparse_matrix=training_matrix)
        self.save_sparse_matrix(file_name=testing_dst, sparse_matrix=test_matrix)

    def remap_dataset(self, path_to_movies, path_to_ratings, target_filepath):
        csv_delimeter = ','
        movie_mapping = defaultdict(int)
        movie_headers = ['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members']
        movie_data = pandas.read_csv(path_to_movies, sep=csv_delimeter, names=movie_headers, header=None, skiprows=1)

        # Remap each movie ID to it's row number in the csv
        for index, row in movie_data.iterrows():
            movie_mapping[row[0]] = index + 2

        rating_headers = ['user_id', 'anime_id', 'rating']
        rating_data = pandas.read_csv(path_to_ratings, sep=csv_delimeter, names=rating_headers, header=None, skiprows=1)
        # Remove any rows that contain a NaN in them
        rating_data.dropna(how='any')
        # Remove any rows that have a rating of -1
        rating_data = rating_data[rating_data.rating != -1]

        with open(target_filepath, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['user_id', 'anime_id', 'rating'])
            # Remap the movie rating file based on the new movie ID
            count = 0
            for index, row in rating_data.iterrows():
                count = count + 1
                if count % 100000 == 0:
                    print "Current Cycle: {} \nTime for last 100000 cycles: {}".format(count, time.time() - tick)
                if count % 100000 == 1:
                    tick = time.time()
                # Check that the anime_id from the rating file exists as a movie in the movie file
                movie_id_new = movie_mapping[row['anime_id']]
                if movie_id_new != 0:
                    user_id = row['user_id']
                    rating = row['rating']
                    writer.writerow([user_id, movie_id_new, rating])

    def run_random_split(self):
        start_time = time.time()
        print "Running random split"
        self.random_split(rating_data=self.rating_data, training_file_name=self.random_training_filepath,
                          test_file_name=self.random_testing_filepath)
        print "Arbitrary split finished in {} seconds".format(time.time() - start_time)

    def run_arbitrary_split(self):
        start_time = time.time()
        print "Running arbitrary split"
        self.arbitrary_split(rating_data=self.rating_data, training_file_name=self.arbitrary_training_filepath,
                          test_file_name=self.arbitrary_testing_filepath)
        print "Arbitrary split finished in {} seconds".format(time.time() - start_time)


if __name__ == '__main__':
    start_whole = time.time()
    preprocess = DataPreprocessing()
    preprocess.run_arbitrary_split()
    preprocess.run_random_split()
    end = time.time()
    print "Time to run program " + str((end - start_whole))