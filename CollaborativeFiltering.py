import numpy as np
import scipy.sparse as sparse
import time
from MatrixOperations import center_matrix
import itertools
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
import math
import os


class CollaborativeFiltering:
    def __init__(self, dataset='random'):
        start_time = time.time()
        print "Initializing Collaborative Filtering"
        training_filepath = 'matrices/{}_training.npz'.format(dataset)
        testing_filepath = 'matrices/{}_test.npz'.format(dataset)
        centered_training_filepath = 'matrices/{}_centered_training.npz'.format(dataset)
        similarity_filepath = 'matrices/{}_similarity.npz'.format(dataset)

        self.training_matrix_coo = sparse.load_npz(training_filepath)
        self.test_matrix_coo = sparse.load_npz(testing_filepath)
        self.training_matrix_csr = self.training_matrix_coo.tocsr()
        self.test_matrix_csr = self.test_matrix_coo.tocsr()
        self.training_matrix_csc = self.training_matrix_coo.tocsc()
        self.test_matrix_csc = self.test_matrix_coo.tocsc()
        self.baseline_rating = self.calculate_global_baseline_rating(self.training_matrix_coo)
        self.relative_movie_average = self.calculate_relative_mean_movie_rating(self.baseline_rating,
                                                                                self.training_matrix_csr)
        self.relative_user_rating = self.calculate_mean_relative_user_rating(self.baseline_rating, self.training_matrix_csc)
        if not os.path.isfile(centered_training_filepath):
            print "Running Center Matrix"
            tick = time.time()
            center_matrix(self.training_matrix_coo,centered_training_filepath,self.relative_movie_average)
            print "Center Matrix done in {} seconds".format(time.time()-tick)
        if os.path.isfile(similarity_filepath):
            print "Loading Similarity Matrix"
            tick = time.time()
            self.cosine_similarity_csr = sparse.load_npz(similarity_filepath)
            print "Load Similarity Matrix done in {} seconds".format(time.time() - tick)
        else:
            print "Building Similarity Matrix"
            tick = time.time()
            self.cosine_similarity_csr=self.fill_similarity_matrix(centered_training_filepath,similarity_filepath)
            print "Building Similarity Matrix done in {} seconds".format(time.time() - tick)
        print "Class initialization done in {} seconds".format(time.time()-start_time)
    def calculate_baseline_estimate(self, movie_ratings, user_ratings, global_movie_mean, coo_matrix):
        baseline_estimate_rating = {}
        # Loop through each entry in the test dataset
        for movie, user, true_rating in itertools.izip(coo_matrix.row, coo_matrix.col,
                                                       coo_matrix.data):
            # Get the baseline rating for this movie in the test set
            movie_baseline = movie_ratings[movie]
            # Get the baseline rating for this user in the test set
            user_baseline = user_ratings[user]
            estimated_rating = movie_baseline + user_baseline + global_movie_mean
            baseline_estimate_rating[(movie, user)] = estimated_rating
        return baseline_estimate_rating

    def calculate_relative_mean_movie_rating(self, global_movie_mean, csr_matrix):
        relative_movie_ratings = {}
        # Calculate the mean of each movie
        movie_sums = csr_matrix.sum(axis=1)
        # Calculate the number of ratings for each movie
        movie_rating_counts = csr_matrix.getnnz(axis=1)
        # Loop through each movie
        number_of_movies = csr_matrix.shape[0]
        for index in xrange(1, number_of_movies):
            # Check to see if the movie has not been rated
            if movie_sums[index] != 0:

                movie_average = float(movie_sums[index]) / movie_rating_counts[index]
                relative_movie_ratings[index] = movie_average - global_movie_mean
            else:
                relative_movie_ratings[index] = 0

        return relative_movie_ratings
    # COO sparse matrix
    def calculate_global_baseline_rating(self,coo_matrix):
        summed_movie_rating = 0
        for i, j, v in itertools.izip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            summed_movie_rating = summed_movie_rating + v
        number_of_ratings = coo_matrix.nnz
        mean_movie_rating = float(summed_movie_rating) / number_of_ratings
        return mean_movie_rating


    def calculate_mean_relative_user_rating(self, global_movie_mean, csc_matrix):
        user_ratings = {}
        # Calculate the mean of each user
        user_sums = csc_matrix.sum(axis=0)
        # Reshape the matrix to array form for proper indexing
        user_sums = user_sums.reshape((user_sums.size, 1))
        # Calculate the number of ratings for each user
        user_rating_counts = csc_matrix.getnnz(axis=0)
        # Loop through each user
        number_of_users = self.training_matrix_csc.shape[1]
        for index in xrange(1, number_of_users):
            # Check to see if the user has not been rated
            if user_sums[index] != 0:
                user_average = float(user_sums[index]) / user_rating_counts[index]
                user_ratings[index] = user_average - global_movie_mean
            else:
                user_ratings[index] = 0

        return user_ratings

    def fill_similarity_matrix(self, src_filename, dst_filename):
        self.centered_training_coo = sparse.load_npz(src_filename)
        similarities_sparse = sk_cosine_similarity(self.centered_training_coo.tocsr(), dense_output=False)
        sparse.save_npz(dst_filename, similarities_sparse)
        return similarities_sparse

    def calculate_error_test(self, estimated_rating, true_rating):
        # Square the error
        error = math.pow(true_rating - estimated_rating, 2)
        return error


    def collab_RMSE(self):
        print "Collaborative Filter RMSE"
        summed_error = 0
        count = 0
        # Loop through each entry in the test dataset
        # Anime dataset has about 2 million test samples
        for movie, user, true_rating in itertools.izip(self.test_matrix_coo.row, self.test_matrix_coo.col, self.test_matrix_coo.data):
            count = count + 1
            if count % 1000 == 0:
                print "Current Cycle: {} \nTime for last 1000 cycles: {}".format(count,time.time()-tick)
            if count % 1000 == 1:
                tick = time.time()
            tock = time.time()
            estimated_rating = self.predict_rating(movie, user)
            #if count <= 10:
            #   print "Time to estimate rating {}".format((time.time()-tock))
            # Calculate the error between the predicted rating and the true rating
            summed_error = summed_error + self.calculate_error_test(estimated_rating, true_rating)
        # Calculate the number of entries in the test set
        test_dataset_size = self.test_matrix_coo.nnz
        # Compute the RMSE on the test set
        rmse = math.sqrt(float(summed_error) / test_dataset_size)
        return rmse


    def predict_rating(self, item, user):
        globalmean = self.baseline_rating
        usermean = self.relative_user_rating[user]
        moviemean=self.relative_movie_average[item]
        bxi = globalmean + usermean + moviemean
        # get nearest neighbors
        item_list, similarity_vector = self.select_nearest(item,user)
        sum_similarity = sum(similarity_vector)
        weighted_rating = 0
        # Use len of itemlist since it may contain fewer than expected elements
        # Depends on how many movies the user has rated
        for ix in xrange(len(item_list)):
            movie = item_list[ix]
            bxj = globalmean + self.relative_user_rating[user] + self.relative_movie_average[movie]
            rxj = self.training_matrix_csr[movie, user]
            weighted_rating = weighted_rating+ (rxj - bxj) * similarity_vector[ix]
        if sum_similarity == 0:
            pred_rating = bxi
        else:
            pred_rating = weighted_rating/sum_similarity + bxi
        return pred_rating

    def collaborative_filter(self):
        tick = time.time()
        RMSE = self.collab_RMSE()
        print "Item-Item Collaborative Filtering RMSE: {}".format(RMSE)
        print "Elapsed Time: {}".format(time.time() - tick)
        return None

    def select_nearest(self,reference_movie,user, k=5):
        similarity_matrix = self.cosine_similarity_csr
        full_similarity_vector = np.zeros((similarity_matrix.shape[0]))
        similarity_row = similarity_matrix.getrow(reference_movie)
        rated_movies = self.training_matrix_csc.getcol(user)
        item_list = np.intersect1d(rated_movies.indices,similarity_row.indices)
        for index in item_list:
            full_similarity_vector[index] =similarity_row[0,index]
        C = len(item_list)
        if C > 5:
            top_items = np.argpartition(full_similarity_vector,-k)[-k:]
        else:
            top_items = item_list
        similarity_vector = full_similarity_vector[top_items]

        return top_items, similarity_vector

if __name__ == '__main__':
    # This should not take more time than a minute or two
    start_time = time.time()
    print "Running Collaborative Filtering on Random Dataset"
    recommender = CollaborativeFiltering()

    # This took two hours to run on a 4 GHz i7e six core processor with 16 GB ram on an SSD
    recommender.collaborative_filter()
    print "Collaborative Filter on Random Dataset done in {} seconds".format(time.time() - start_time)
    start_time = time.time()
    print "Running Collaborative Filtering on Arbitrary Dataset"
    recommender_arbitary = CollaborativeFiltering(dataset='arbitrary')
    recommender_arbitary.collaborative_filter()
    print "Collaborative Filter on Arbitrary Dataset done in  {} seconds".format(time.time() - start_time)
