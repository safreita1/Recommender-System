from numpy import genfromtxt
import scipy.sparse as sparse
import numpy as np
import csv
import math
import time
import itertools
import pandas

def write_matrix_to_csv(user_col, movie_row, rating_data, file_name):
    with open(file_name, 'wb') as file:
        writer = csv.writer(file)
        for ix in xrange(len(user_col)):
            writer.writerow((user_col[ix], movie_row[ix], rating_data[ix]))

def calculate_baseline_RMSE(test_matrix, movie_ratings, user_ratings, global_movie_mean):
    summed_error = 0

    # Loop through each entry in the test dataset
    for movie, user, true_rating in zip(test_matrix.row, test_matrix.col, test_matrix.data):
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
        summed_error = summed_error + calculate_error_test(movie_baseline, user_baseline, global_movie_mean, true_rating)

    # Calculate the number of entries in the test set
    test_dataset_size = test_matrix.nnz

    # Compute the RMSE on the test set
    rmse = math.sqrt(float(summed_error) / test_dataset_size)

    return rmse

def calculate_error_test(movie_baseline, user_baseline, global_movie_mean, true_rating):
    estimated_rating = movie_baseline + user_baseline + global_movie_mean

    # Square the error
    error = math.pow(true_rating - estimated_rating, 2)
    return error

def calculate_global_mean_movie_rating(training_matrix):
    summed_movie_rating = 0
    for i,j,v in itertools.izip(training_matrix.row, training_matrix.col, training_matrix.data):
        summed_movie_rating = summed_movie_rating + v

    number_of_ratings = training_matrix.nnz
    mean_movie_rating = float(summed_movie_rating) / number_of_ratings

    return mean_movie_rating

def calculate_mean_movie_rating(training_matrix, global_movie_mean):
    movie_ratings = {}
    for row in training_matrix.row:
        row_data = training_matrix.getrow(row)

        sum_movie = row_data.sum()
        number_movie = row_data.nnz
        movie_average = float(sum_movie) / number_movie

        movie_ratings[row] = movie_average - global_movie_mean

    return movie_ratings

def calculate_mean_user_rating(training_matrix, global_movie_mean):
    user_ratings = {}
    for col in training_matrix.col:
        col_data = training_matrix.getcol(col)

        sum_user = col_data.sum()
        number_users = col_data.nnz
        user_average = float(sum_user) / number_users

        user_ratings[col] = user_average - global_movie_mean

    return user_ratings

# Remove any entries from the dataset that don't correspond to a rating
def preprocess_data(file_name):
    user_entries = []
    movie_entries = []
    rating_entries = []
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
def create_training_test_split():
    # Load data in from dataset
    rating_data = genfromtxt('preprocessed_dataset.csv', delimiter=',')

    # Find the 80-20 split point from the loaded data
    split_delimiter = int(.8 * len(rating_data[:,0]))

    # 80% of the data goes to the training set
    training_user_col = rating_data[:split_delimiter,0]
    training_movie_row = rating_data[:split_delimiter,1]
    training_rating_data = rating_data[:split_delimiter,2]

    # Write the training data to a csv file
    write_matrix_to_csv(training_user_col, training_movie_row, training_rating_data, 'training_data.csv')

    # 20% of the data goes to the test set
    test_user_col = rating_data[split_delimiter:,0]
    test_movie_row  = rating_data[split_delimiter:,1]
    test_rating_data = rating_data[split_delimiter:,2]

    # Write the test data to a csv file
    write_matrix_to_csv(test_user_col, test_movie_row, test_rating_data, 'test_data.csv')

    training_matrix = sparse.coo_matrix((training_rating_data, (training_movie_row, training_user_col)), shape=(40000, 73516))
    test_matrix = sparse.coo_matrix((test_rating_data, (test_movie_row, test_user_col)), shape=(40000, 73516))

    return training_matrix, test_matrix

def save_sparse_matrix(file_name, sparse_matrix):
    sparse.save_npz(file_name, sparse_matrix)

def load_sparse_matrix(file_name):
    return sparse.load_npz(file_name)



if __name__ == '__main__':
    start_whole = time.time()
    load_matrix = False
    training_matrix = None
    test_matrix = None

    start = time.time()

    # Load the sparse matrix from a file
    if load_matrix:
        training_matrix = load_sparse_matrix('training_matrix.npz')
        test_matrix = load_sparse_matrix('test_matrix.npz')
    # Process the dataset, load it into a sparse matrix, save the sparse matrix into a file
    else:
        preprocess_data('rating.csv')
        training_matrix, test_matrix = create_training_test_split()
        save_sparse_matrix('training_matrix.npz', training_matrix)
        save_sparse_matrix('test_matrix.npz', test_matrix)

    end = time.time()
    print "Time to load data"
    print(end - start)

    start = time.time()

    global_movie_mean = calculate_global_mean_movie_rating(training_matrix)

    end = time.time()
    print "Time to calculate global mean"
    print(end - start)

    start = time.time()

    movie_ratings = calculate_mean_movie_rating(training_matrix, global_movie_mean)
    user_ratings = calculate_mean_user_rating(training_matrix, global_movie_mean)

    end = time.time()
    print "Time to calculate user and movie mean"
    print(end - start)

    start = time.time()

    rmse = calculate_baseline_RMSE(test_matrix, movie_ratings, user_ratings, global_movie_mean)

    end = time.time()
    print "Time to calculate rmse"
    print(end - start)

    print rmse

    end = time.time()
    print "Time to run program"
    print(end - start_whole)


### Things to do: OO code, scale to full dataset,  better training/test split

