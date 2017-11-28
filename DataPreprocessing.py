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
        self.training_matrix = None
        self.test_matrix = None

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
    #Input:
    # data: 1d array or np array
    def count_unique(self, data):
        count = 0
        seen = []
        for element in data:
            if element not in seen:
                count = count + 1
                seen.append(element)
        return count
    #Input:
    # data: numpy array
    def random_sample(self, data, seed=42):
       # print "init random sample"
        random.seed(seed)
        percent_test = 0.2
        percent_val = 0.1

        total_size = data.shape[0]
        test_val_size = int(total_size * (percent_val +percent_test ))
        #print "generate test indices"
        #tick = time.time()
        test_list = random.sample(xrange(total_size),test_val_size)
        #print time.time()-tick

       # print "create test matrix"
       # print "Build and populate test data"
        test_size = int(test_val_size*2/3)+1
        val_size = test_val_size-test_size
        test_data = np.zeros((test_size, 3))
        val_data = np.zeros((val_size,3))
        # Populate test matrix
        test_index = 0
        val_index = 0
        for ix in test_list:
            if val_index < val_size:
                val_data[val_index]= data[ix]
                val_index = val_index + 1
            else:
                test_data[test_index]=  data[ix]
                test_index = test_index + 1


       # print "Populate training data"
        training_data = np.delete(data, test_list, 0)
        return training_data, test_data, val_data



    # Input:
    # rating_data: numpy array
    def random_sparse_split(self, rating_data, training_file_name, test_file_name,val_file_name):
        # Randomly split the data into an 80-20 training/test set respectively
        training_data, test_data, val_data = self.random_sample(rating_data)
        # Training data
        training_user_col = training_data[:, 0]
        training_movie_row = training_data[:, 1]
        training_rating_data = training_data[:, 2]
        # Write the training data to a csv file
        self.write_matrix_to_csv(training_user_col, training_movie_row, training_rating_data, training_file_name)
        # Test data
        test_user_col = test_data[:, 0]
        test_movie_row = test_data[:, 1]
        test_rating_data = test_data[:, 2]
        # Write the test data to a csv file
        self.write_matrix_to_csv(test_user_col, test_movie_row, test_rating_data, test_file_name)
        # Find the sparse matrix dimensions
        sparse_user_size = max(rating_data[:, 0]) + 1
        sparse_movie_size = max(rating_data[:, 1]) + 1
        val_user_col = val_data[:, 0]
        val_movie_row = val_data[:, 1]
        val_rating_data = val_data[:, 2]
        self.write_matrix_to_csv(val_user_col, val_movie_row, val_rating_data, val_file_name)
        self.validation_matrix = sparse.coo_matrix((val_rating_data, (val_movie_row, val_user_col)),
                                            shape=(sparse_movie_size, sparse_user_size), dtype=np.float64)
        self.training_matrix = sparse.coo_matrix((training_rating_data, (training_movie_row, training_user_col)),
                                            shape=(sparse_movie_size, sparse_user_size), dtype=np.float64)
        self.test_matrix = sparse.coo_matrix((test_rating_data, (test_movie_row, test_user_col)),
                                             shape=(sparse_movie_size,sparse_user_size),dtype=np.float64)

    def arbitrary_sparse_split(self, rating_data, training_file_name, test_file_name):
        # Find the 80-20 split point from the loaded data
        split_delimiter = int(.8 * len(rating_data[:, 0]))
        # 80% of the data goes to the training set
        training_user_col = rating_data[:split_delimiter, 0]
        training_movie_row = rating_data[:split_delimiter, 1]
        training_rating_data = rating_data[:split_delimiter, 2]
        # Write the training data to a csv file
        self.write_matrix_to_csv(training_user_col, training_movie_row, training_rating_data, training_file_name)
        # 20% of the data goes to the test set
        test_user_col = rating_data[split_delimiter:, 0]
        test_movie_row = rating_data[split_delimiter:, 1]
        test_rating_data = rating_data[split_delimiter:, 2]
        # Write the test data to a csv file
        self.write_matrix_to_csv(test_user_col, test_movie_row, test_rating_data, test_file_name)
        # Find the sparse matrix dimensions
        sparse_user_size = max(rating_data[:, 0])+1
        sparse_movie_size = max(rating_data[:, 1])+1
        # Create the sparse matrices for the training and test data
        self.training_matrix = sparse.coo_matrix((training_rating_data, (training_movie_row, training_user_col)),
                                                 shape=(sparse_movie_size, sparse_user_size), dtype=np.float64)
        self.test_matrix = sparse.coo_matrix((test_rating_data, (test_movie_row, test_user_col)),
                                             shape=(sparse_movie_size,sparse_user_size),dtype=np.float64)

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
            for index, row in rating_data.iterrows():
                # Check that the anime_id from the rating file exists as a movie in the movie file
                movie_id_new = movie_mapping[row['anime_id']]
                if movie_id_new != 0:
                    user_id = row['user_id']
                    rating = row['rating']
                    writer.writerow([user_id, movie_id_new, rating])



if __name__ == '__main__':
    arbitrary_training_filepath = 'csv/arbitrary_training.csv'
    arbitrary_testing_filepath = 'csv/arbitrary_test.csv'
    random_training_filepath = 'csv/random_training.csv'
    random_testing_filepath = 'csv/random_test.csv'
    random_validation_filepath = 'csv/rantom_val.csv'
    remapped_filepath = 'csv/remapped_rating.csv'
    movie_filepath = 'csv/anime.csv'
    rating_filepath = 'csv/rating.csv'

    init_all = True
    start_whole = time.time()
    preprocess = DataPreprocessing()

    # Remap dataset once
    if not os.path.isfile(remapped_filepath):
        preprocess.remap_dataset(path_to_movies=movie_filepath, path_to_ratings=rating_filepath, target_filepath=remapped_filepath)

    # Load remapped dataset
    if os.path.isfile(remapped_filepath):
        rating_data = preprocess.load_csv_as_nparray(remapped_filepath)
    else:
        print "Insufficient data"
        exit(1)

    # Create arbitrary training/test split once, save as csv files
    if not os.path.isfile(arbitrary_training_filepath) or not os.path.isfile(arbitrary_testing_filepath):
        preprocess.arbitrary_sparse_split(rating_data, arbitrary_training_filepath, arbitrary_testing_filepath)
        # Save arbitrarily sampled sparse matrices as npz files
        preprocess.save_sparse_matrix(file_name='matrices/arbitrary_training', sparse_matrix=preprocess.training_matrix)
        preprocess.save_sparse_matrix(file_name='matrices/arbitrary_test', sparse_matrix=preprocess.test_matrix)

    # Create randomly sampled training/test split once, save as csv files
    if not os.path.isfile(random_training_filepath) or not os.path.isfile(random_testing_filepath) or not \
            os.path.isfile(random_validation_filepath):
        preprocess.random_sparse_split(rating_data=rating_data, training_file_name=random_training_filepath,
                                   test_file_name=random_testing_filepath,val_file_name=random_validation_filepath)

         #Save randomly sampled sparse matrices as npz files
        preprocess.save_sparse_matrix(file_name='matrices/random_training', sparse_matrix=preprocess.training_matrix)
        preprocess.save_sparse_matrix(file_name='matrices/random_test', sparse_matrix=preprocess.test_matrix)
        preprocess.save_sparse_matrix(file_name='matrices/random_val', sparse_matrix=preprocess.validation_matrix)

    end = time.time()
    print "Time to run program " + str((end - start_whole))