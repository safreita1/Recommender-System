import numpy as np
import scipy.sparse as sparse
import time
import scipy
from MatrixOperations import center_matrix
import itertools
import math
import os


class CollaborativeFiltering:
    def __init__(self, dataset='random'):
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
            center_matrix(self.training_matrix_coo,centered_training_filepath,self.relative_movie_average)
        if os.path.isfile(similarity_filepath):
            self.cosine_similarity_csr = sparse.load_npz(similarity_filepath)
        else:
            self.cosine_similarity_csr=self.fill_similarity_matrix(centered_training_filepath,similarity_filepath)

        #self.cosine_similarity_matrix_csr = sparse.load_npz(similarity_filepath).tocsr

        #self.baseline_estimate_rating = self.calculate_baseline_estimate(self.relative_movie_average,
        #                                                                 self.baseline_user_rating,
        #                                                                 self.baseline_rating,
        #                                                                 self.
        #                                                                 )

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
                # TODO Evaluate if zero is best here
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

    # Assume items already centered
    def cosine_similarity(self, item1, item2):
        dotprod = np.dot(item1, item2)
        mag1 = np.linalg.norm(item1)
        mag2 = np.linalg.norm(item2)
        if mag1 == 0 or mag2 == 0:
            return 0
        similarity = dotprod / (mag1 * mag2)
        if similarity > 2:
            print "Dotprod {} mag1 {} mag2 {} similarity {}".format(dotprod,mag1,mag2,similarity)
        return similarity


    # For use in Cosine similarity
    def create_full_movie_vector(self, movie):
        size = self.training_matrix_csr.shape[1]
        vector = np.zeros(size)
        row = self.training_matrix_csr.getrow(movie)
        for index in row.indices:
            value = self.training_matrix_csr[movie, index]
            vector[index] = value
        return vector


    def fill_similarity_matrix(self, src_filename, dst_filename):
        self.centered_training_coo = sparse.load_npz(src_filename)
        num_movies = self.centered_training_coo.shape[0]
        self.cosine_similarity_matrix = np.zeros((num_movies + 1, num_movies + 1))
        # Build a list of numpy arrays to make retrieval less costly
        tick = time.time()
        movie_rows = self.centered_training_coo.toarray()

        print "Time to build movie list: {}".format(time.time() - tick)
        # Iterate over all movies, for all other movies, and compute cosine similarity
        for movie1 in xrange(1, num_movies):
            tick = time.time()
            item1 = movie_rows[movie1]
            for movie2 in xrange(1, num_movies):
                if movie1 == movie2:
                    cosine_similarity = 1
                else:
                    item2 = movie_rows[movie2]
                    cosine_similarity = self.cosine_similarity(item1, item2)
                    # movie_matrix[(movie1, movie2)] =
                t1 = self.cosine_similarity_matrix[movie1, movie2]
                self.cosine_similarity_matrix[movie1, movie2] = cosine_similarity
                self.cosine_similarity_matrix[movie2, movie1] = cosine_similarity
            if movie1 % 1000 == 0:
                print "Iteration {} took {} seconds".format(movie1, time.time() - tick)
        np.savetxt("csv/similarity_matrix_random.csv", self.cosine_similarity_matrix, delimiter=",")
        sparse_matrix = sparse.coo_matrix(self.cosine_similarity_matrix)
        sparse.save_npz(dst_filename, sparse_matrix)
        print "done"
        return sparse_matrix

    def calculate_error_test(self, estimated_rating, true_rating):

        # Square the error
        error = math.pow(true_rating - estimated_rating, 2)
        return error


    def collab_RMSE(self):
        summed_error = 0
        count = 0
        # Loop through each entry in the test dataset
        for movie, user, true_rating in itertools.izip(self.test_matrix_coo.row, self.test_matrix_coo.col, self.test_matrix_coo.data):
            count = count + 1
            if count % 1000 == 0:
                print "{} cycle time per 1000: {}".format(count,time.time()-tick)
            if count % 1000 == 1:
                tick = time.time()
            estimated_rating = self.predict_rating(movie, user)
            # Calculate the error between the predicted rating and the true rating
            summed_error = summed_error + self.calculate_error_test(estimated_rating, true_rating)

        # Calculate the number of entries in the test set
        test_dataset_size = self.test_matrix_coo.nnz

        # Compute the RMSE on the test set
        rmse = math.sqrt(float(summed_error) / test_dataset_size)

        return rmse

    def predict_rating(self, item, user):
        # get nearest neighbors
        globalmean = self.baseline_rating
        usermean = self.relative_user_rating[user]
        moviemean=self.relative_movie_average[item]
        bxi = globalmean + usermean + moviemean
        item_list, similarity_vector = self.select_nearest(item,user)
        sum_similarity = sum(similarity_vector)
        weighted_rating = 0
        # Use len of itemlist since it may contain fewer than expected elements
        # Depends on how many movies the user has rated
        #print len(item_list)
        for ix in xrange(len(item_list)):
            movie = item_list[ix]
            bxj = globalmean + self.relative_user_rating[user] + self.relative_movie_average[movie]
            rxj = self.training_matrix_csr[movie, user]
            weighted_rating = weighted_rating+ (rxj - bxj) * similarity_vector[ix]
        if sum_similarity == 0:
            return bxi
        else:
            pred_rating = weighted_rating/sum_similarity + bxi
        return pred_rating

    # Stub
    def collaborative_filter(self):
        shortened_matrix_filepath = 'matrices/shortened_similarity.npz'
        tick = time.time()
        if os.path.isfile(shortened_matrix_filepath):
            self.shortened_cosine_similarity = sparse.load_npz(shortened_matrix_filepath)
        else:
            self.shortened_cosine_similarity = self.erase_all_but_top_n(self.cosine_similarity_csr,
                                                                        shortened_matrix_filepath)

        RMSE = self.collab_RMSE()
        # self.predict_rating(10, 10)
        print "Item-Item Collaborative Filtering RMSE: {}".format(RMSE)
        print "Elapsed Time: {}".format(time.time() - tick)
        return None

        #Completely untested
    def select_nearest(self,reference_movie,user, k=5):
        similarity_matrix = self.shortened_cosine_similarity
        similarity_vector = np.zeros((k,1))
        similarity_row = similarity_matrix.getrow(reference_movie)
        #print similarity_row#[reference_movie,:]
        rated_movies = self.training_matrix_csc.getcol(user)
        item_list = []
        for index in rated_movies.indices:
            if index in similarity_row.indices:
                item_list = self.top_n(item_list,index,similarity_row,k)
        #item_list = zip(*heapq.nlargest(k, enumerate(row), key=operator.itemgetter(1)))[0]

        for ix in xrange(len(item_list)):
            similarity_vector[ix] = similarity_row[0,item_list[ix]]
        return item_list, similarity_vector

    def top_n(self, itemlist, item, sparserow, n):
        count_larger = 0
        dict_larger = {}
        if item in sparserow.indices:
            rating = sparserow[0,item]
            for index in itemlist:
                comp_rating = sparserow[0,index]
                dict_larger[index] = comp_rating
                if comp_rating >= rating:
                    #dict_larger[index] = comp_rating
                    count_larger = count_larger + 1
        else:
            print "Item {} not in sparse row".format(item)
        if count_larger < n:
            if len(itemlist) < 5:
                itemlist.append(item)
            else:
                displaced = min(dict_larger,key=dict_larger.get)
                replace_index = itemlist.index(displaced)
                itemlist[replace_index] = item

        # Else item not added
        return itemlist

    # best if used with csr
    def erase_all_but_top_n(self, sparsematrix, filepath, n=20):
        nrows = sparsematrix.shape[0]
        #print nrows
        new_matrix = np.zeros((20 * nrows, 3))
        new_matrix_counter = 0
        for row in xrange(1, nrows):
            tick = time.time()
            sparserow = sparsematrix.getrow(row)
            # Store (index, value) in list
            top_n = []
            for movie in sparserow.indices:
                if movie == row:
                    continue
                if len(top_n) < n:
                    top_n.append((sparserow[0, movie], movie))
                else:
                    test_value = sparserow[0, movie]
                    min_value = float('inf')
                    min_movie = -1
                    min_index = -1
                    for ix, item in enumerate(top_n):
                        comp_value = item[1]
                        comp_index = item[0]
                        if comp_value < min_value:
                            min_value = comp_value
                            min_movie = comp_index
                            min_index = ix
                    if min_index < 0:
                        print "Failed to find smallest"
                    if test_value < min_value:
                        top_n.pop(min_index)
                        top_n.append((movie, test_value))
                        # sparsematrix[row,min_movie] = 0
            for record in top_n:
                new_matrix[new_matrix_counter] = (row, record[0], record[1])
                new_matrix_counter = new_matrix_counter + 1

            print "Iteration {} took {} seconds".format(row, time.time() - tick)
        # sparsematrix.eliminate_zeroes()

        row_movies = new_matrix[0:new_matrix_counter, 0]
        col_movies = new_matrix[0:new_matrix_counter, 1]
        similarity = new_matrix[0:new_matrix_counter, 2]
        sparse_movie_size = nrows
        new_sparse_matrix = sparse.coo_matrix((similarity, (row_movies, col_movies)), shape=(sparse_movie_size,
                                                                                             sparse_movie_size),
                                              dtype=np.float64)
        sparse.save_npz(filepath, new_sparse_matrix)
        return sparsematrix
if __name__ == '__main__':

    start_whole = time.time()
    recommender = CollaborativeFiltering()
    recommender.collaborative_filter()

            # if not os.path.isfile(centered_training_filepath):
    #     tick = time.time()
    #     recommender.center_matrix(recommender.training_matrix_coo,centered_training_filepath)
    #     print "Time to center matrix: {}".format(time.time()-tick)
    #     #recommender.test_center()