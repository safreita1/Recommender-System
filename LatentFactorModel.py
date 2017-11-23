from scipy.sparse.linalg import svds
import itertools
import numpy as np
import math
from collections import defaultdict
from MatrixOperations import convert_coo_to_csc_and_csr, center_matrix_user
import time
import os
import datetime


class LatentFactorModel:
    def __init__(self, epochs, k, learning_rate, lambda_reg, training_coo, test_coo, user_average):
        self.P = None
        self.Q = None
        self.epochs = epochs
        self.k = k
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.training_coo = training_coo
        self.training_csc = None
        self.training_csr = None
        self.test_coo = test_coo
        self.test_csc = None
        self.test_csr = None
        self.user_average = user_average
        self.user_std = defaultdict(int)

    def calculate_user_std(self):
        for movie, user, rating in itertools.izip(self.training_coo.row, self.training_coo.col, self.training_coo.data):
            self.user_std[user] = self.user_std[user] + math.pow((self.user_average[user] - rating), 2)

        number_of_users = len(self.user_std)
        for user, value in self.user_std.iteritems():
            self.user_std[user] = math.sqrt(self.user_std[user] / (number_of_users - 1))


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
        directory = 'optimization/{}/epoch_{}/'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), epoch)
        if not os.path.exists(directory):
            os.makedirs(directory)

            p_matrix = "{}{}".format(directory, "P.npy")
            q_matrix = "{}{}".format(directory, "Q.npy")

            np.save(arr=self.P, file=p_matrix)
            np.save(arr=self.Q, file=q_matrix)

            rmse_file = directory + "RMSE.txt"
            rmse_info = 'RMSE Training: {} \n RMSE Test: {}'.format(rmse_training, rmse_test)
            f = open(rmse_file, "w+")
            f.write(rmse_info)
            f.close()
        else:
            print "Error: directory already exists"

        hyper_param_file = 'optimization/hyperparams.txt'
        params = 'Learning rate: {} \nRegularization rate: {} \nNumber of factors (k): {} \n# of epochs: {}'.format(
            self.learning_rate, self.lambda_reg, self.k, self.epochs)
        f = open(hyper_param_file, "w+")
        f.write(params)
        f.close()


    def optimize_matrices(self):
        for epoch in xrange(self.epochs):

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

            self.save_model(epoch=epoch, rmse_test=rmse_test, rmse_training=rmse_training)
            print "Epoch {} model saved".format(epoch)

            count = 0
            start = time.time()
            print "Movie 4830, user 47914, true rating: 6. Predicted rating: " + str(self.predicted_value(4830, 47914) + self.user_average[47914])

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


    def run_model(self):
        self.training_coo = center_matrix_user(sparse_matrix=self.training_coo, user_average=self.user_average)

        self.training_csc, self.training_csr = convert_coo_to_csc_and_csr(self.training_coo)
        self.test_csc, self.test_csr = convert_coo_to_csc_and_csr(self.test_coo)

        self.run_svd()
        self.optimize_matrices()