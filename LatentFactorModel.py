from scipy.sparse.linalg import svds
import itertools
import numpy as np
import math
from collections import defaultdict
from MatrixOperations import convert_coo_to_csc_and_csr, center_matrix_user
import time
import os
import datetime
from scipy import sparse


class LatentFactorModel:
    def __init__(self, epochs, k, learning_rate, lambda_reg):
        # Load the sparse matrix from a file
        self.training_filepath = 'matrices/{}_training.npz'.format('random')
        self.testing_filepath = 'matrices/{}_test.npz'.format('random')
        self.training_coo = self.load_sparse_matrix(self.training_filepath)
        self.test_coo = self.load_sparse_matrix(self.testing_filepath)

        self.P = None
        self.Q = None
        self.epochs = epochs
        self.current_epoch = 0
        self.k = k
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.training_csc = None
        self.training_csr = None
        self.test_csc = None
        self.test_csr = None
        self.user_average = {}
        self.global_mean = 0.0
        self.model_directory = None
        self.model_loaded = False

        self.training_csc, self.training_csr = convert_coo_to_csc_and_csr(self.training_coo)
        self.test_csc, self.test_csr = convert_coo_to_csc_and_csr(self.test_coo)

        self.calculate_mean_user_rating()
        self.training_coo = center_matrix_user(sparse_matrix=self.training_coo, user_average=self.user_average)

        # Recalculate the CSC and CSR matrices after centering
        self.training_csc, self.training_csr = convert_coo_to_csc_and_csr(self.training_coo)
        self.test_csc, self.test_csr = convert_coo_to_csc_and_csr(self.test_coo)


    def load_sparse_matrix(self, file_name):
        return sparse.load_npz(file_name)

    def calculate_global_baseline_rating(self):
        summed_movie_rating = 0
        for i, j, v in itertools.izip(self.training_coo.row, self.training_coo.col, self.training_coo.data):
            summed_movie_rating = summed_movie_rating + v

        number_of_ratings = self.training_coo.nnz
        self.global_mean = float(summed_movie_rating) / number_of_ratings

    def calculate_mean_user_rating(self):
        self.calculate_global_baseline_rating()

        # Calculate the mean of each user
        user_sums = self.training_csc.sum(axis=0)
        # Reshape the matrix to array form for proper indexing
        user_sums = user_sums.reshape((user_sums.size, 1))
        # Calculate the number of ratings for each user
        user_rating_counts = self.training_csc.getnnz(axis=0)

        # Loop through each user
        number_of_users = self.training_csc.shape[1]
        for index in xrange(1, number_of_users):
            # Check to see if the user has not rated
            if user_sums[index] != 0:
                user_average = float(user_sums[index]) / user_rating_counts[index]
                self.user_average[index] = user_average
            else:
                self.user_average[index] = self.global_mean

    def run_svd(self):
        u, s, vt = svds(self.training_csc, k=self.k)

        self.Q = u
        diag_matrix = np.diag(s)
        self.P = diag_matrix.dot(vt)

    def predicted_value(self, movie, user):
        col = self.P[:, user]
        row = self.Q[movie, :]
        return row.dot(col)

    def error(self, movie, user):
        actual_value = self.training_csr[movie, user]
        predicted_value = self.predicted_value(movie, user)
        return actual_value - predicted_value

    def square_error_train(self, movie, user):
        actual_value = self.training_csr[movie, user]
        predicted_value = self.predicted_value(movie, user) + self.user_average[user]
        return math.pow(actual_value - predicted_value, 2)

    def square_error_test(self, movie, user):
        actual_value = self.test_csr[movie, user]
        predicted_value = self.predicted_value(movie, user) + self.user_average[user]
        return math.pow(actual_value - predicted_value, 2)

    def calculate_test_rmse(self):
        summed_error = 0

        # Loop through each entry in the test dataset
        for movie, user, true_rating in itertools.izip(self.test_coo.row, self.test_coo.col, self.test_coo.data):
            summed_error = summed_error + self.square_error_test(movie, user)

        # Calculate the number of entries in the test set
        test_dataset_size = self.test_coo.nnz

        rmse = math.sqrt(float(summed_error) / test_dataset_size)

        return rmse

    def calculate_training_rmse(self):
        summed_error = 0

        # Loop through each entry in the test dataset
        for movie, user, true_rating in itertools.izip(self.training_coo.row, self.training_coo.col, self.training_coo.data):
            summed_error = summed_error + self.square_error_train(movie, user)

        # Calculate the number of entries in the test set
        training_dataset_size = self.training_coo.nnz

        rmse = math.sqrt(float(summed_error) / training_dataset_size)

        return rmse


    def save_model(self, epoch, rmse_test, rmse_training):
        # Only create the hyperparameter file once
        if epoch == 0:
            self.model_directory = 'optimization/{}/'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            directory = '{}epoch_{}/'.format(self.model_directory, epoch)
            os.makedirs(self.model_directory)

            self.save_hyperparameters()
        else:
            directory = '{}epoch_{}/'.format(self.model_directory, epoch)

        if not os.path.exists(directory):
            os.makedirs(directory)
            self.save_matrices(directory=directory)
            self.save_rmse_file(directory=directory, rmse_training=rmse_training, rmse_test=rmse_test)
        else:
            print "Error: directory already exists"

    def find_current_epoch(self, model_directory):
        highest_epoch = -1
        for directory in os.listdir(model_directory):
            # Check that it's actually a directory
            if os.path.isdir(model_directory + directory):
                temp, current_epoch = directory.split('_')
                current_epoch = int(current_epoch)
                if current_epoch > highest_epoch:
                    highest_epoch = current_epoch

        return highest_epoch


    def load_hyperparameters(self, path_to_hyperparam_file):
        with open(path_to_hyperparam_file) as f:
            lines = f.readlines()
            temp, learning_rate = lines[0].split(':')
            temp, reg_rate = lines[1].split(':')
            temp, num_factors = lines[2].split(':')
            temp, epochs = lines[3].split(':')

            self.learning_rate = float(learning_rate.strip())
            self.lambda_reg = float(reg_rate.strip())
            self.k = int(num_factors.strip())
            self.epochs = int(epochs.strip())

    def save_hyperparameters(self):
        hyper_param_file = '{}hyperparams.txt'.format(self.model_directory)
        params = 'Learning rate: {} \nRegularization rate: {} \nNumber of factors (k): {} \n# of epochs: {}'.format(self.learning_rate, self.lambda_reg, self.k, self.epochs)
        f = open(hyper_param_file, "w+")
        f.write(params)
        f.close()

    def save_rmse_file(self, directory, rmse_training, rmse_test):
        rmse_file = directory + "RMSE.txt"
        rmse_info = 'RMSE Training: {} \nRMSE Test: {}'.format(rmse_training, rmse_test)
        f = open(rmse_file, "w+")
        f.write(rmse_info)
        f.close()

    def save_matrices(self, directory):
        p_matrix = "{}{}".format(directory, "P.npy")
        q_matrix = "{}{}".format(directory, "Q.npy")

        np.save(arr=self.P, file=p_matrix)
        np.save(arr=self.Q, file=q_matrix)

    def load_model(self, model_directory):
        self.model_loaded = True

        # Find the last epoch that was saved
        epoch = self.find_current_epoch(model_directory=model_directory)
        if epoch >= 0:
            self.current_epoch = epoch
        else:
            print "Failed to find epoch folder"
            exit(1)

        path_to_model = model_directory + 'epoch_{}/'.format(epoch)
        path_to_hyperparam = model_directory + 'hyperparams.txt'

        self.model_directory = model_directory

        # Check that hyperparamter file exists
        if os.path.exists(path_to_hyperparam):
            self.load_hyperparameters(path_to_hyperparam)
        else:
            print "Failed to find the hyperparameter file."
            exit(1)

        path_to_matrix_P = path_to_model + 'P.npy'
        path_to_matrix_Q = path_to_model + 'Q.npy'
        # Check that the models exist
        if os.path.exists(path_to_model):
            self.P = np.load(path_to_matrix_P)
            self.Q = np.load(path_to_matrix_Q)
        else:
            print "Failed to load model."
            exit(1)


    def calculate_epoch_error(self, epoch):
        #print "Movie 4830, user 47914, true rating: 6. Predicted rating: " + str( self.predicted_value(4830, 47914) + self.user_average[47914])
        start = time.time()
        rmse_test = self.calculate_test_rmse()
        end = time.time()
        print "Time to calculate RMSE test: {}".format(end - start)

        start = time.time()
        rmse_training = self.calculate_training_rmse()
        end = time.time()
        print "Time to calculate RMSE training: {}".format(end - start)

        print "Training RMSE for epoch {}: {}".format(epoch, rmse_training)
        print "Test RMSE for epoch {}: {}".format(epoch, rmse_test)

        return rmse_test, rmse_training


    def optimize_matrices(self):
        model_already_tested_and_saved = self.model_loaded
        for epoch in xrange(self.current_epoch, self.epochs):

            if not model_already_tested_and_saved:
                rmse_test, rmse_training = self.calculate_epoch_error(epoch)

                self.save_model(epoch=epoch, rmse_test=rmse_test, rmse_training=rmse_training)
                print "Epoch {} model saved".format(epoch)
            else:
                model_already_tested_and_saved = False

            count = 0
            start = time.time()
            # Loop through each entry in the training dataset
            for movie, user, true_rating in itertools.izip(self.training_coo.row, self.training_coo.col, self.training_coo.data):

                if count % 100000 == 0:
                    print "Current count {}".format(count)
                    end = time.time()
                    print "Time taken {}".format(end-start)
                    start = end
                count = count + 1

                # Loop through every latent factor
                for k in xrange(self.k):
                    error = 2 * self.error(movie, user) * self.P[k, user]
                    regularization = - 2 * self.lambda_reg * self.Q[movie, k]
                    gradient_q =  self.learning_rate * (error + regularization)
                    self.Q[movie, k] = self.Q[movie, k] + gradient_q

                    gradient_p = self.learning_rate * (2 * self.error(movie, user) * self.Q[movie, k] - 2 * self.lambda_reg * self.P[k, user])
                    self.P[k, user] = self.P[k, user] + gradient_p

                    # self.Q[movie, k] = self.Q[movie, k] + self.learning_rate * (2 * self.error(movie, user) * self.P[k, user] - 2 * self.lambda_reg * self.Q[movie, k])
                    # self.P[k, user] = self.P[k, user] + self.learning_rate * (2 * self.error(movie, user) * self.Q[movie, k] - 2 * self.lambda_reg * self.P[k, user])


    def run_new_model(self):
        self.run_svd()
        self.optimize_matrices()

    def run_old_model(self, model_directory):
        self.load_model(model_directory=model_directory)
        self.optimize_matrices()