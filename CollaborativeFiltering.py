import numpy as np
import scipy.sparse as sparse
import time
import scipy
import itertools
import math


class CollaborativeFiltering:
    def __init__(self):
        self.training_matrix_coo = None
        self.test_matrix_coo = None
        self.training_matrix_csr = None
        self.test_matrix_csr = None
        self.training_matrix_csc = None
        self.test_matrix_csc = None
        self.baseline_rating = {}
        self.baseline_movie_rating = {}
        self.baseline_user_rating = {}

    # Assume items already centered
    def cosine_similarity(self, item1, item2):
        dotprod = np.dot(item1, item2)
        mag1 = np.linalg.norm(item1)
        mag2 = np.linalg.norm(item2)
        if mag1 == 0 or mag2 == 0:
            return 0
        similarity = dotprod / (mag1 * mag2)
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
            print "Iteration {} took {} seconds".format(movie1, time.time() - tick)
        np.savetxt("csv/similarity_matrix_random.csv", self.cosine_similarity_matrix, delimiter=",")
        sparse_matrix = sparse.coo_matrix(self.cosine_similarity_matrix)
        sparse.save_npz(dst_filename, sparse_matrix)
        print "done"


    # Nonfunctional
    def sparse_cosine_similarity(self, row1, row2):
        dotprod = row1.dot(row2)
        mag1 = scipy.sparse.linalg.norm(row1)
        mag2 = scipy.sparse.linalg.norm(row2)

        similarity = dotprod / (mag1 * mag2)
        if mag1 == 0 or mag2 == 0:
            return 0
        else:
            return similarity

    def collab_RMSE(self):
        summed_error = 0
        count = 0
        # Loop through each entry in the test dataset
        for movie, user, true_rating in itertools.izip(self.test_matrix_coo.row, self.test_matrix_coo.col, self.test_matrix_coo.data):
            count = count + 1
            if count % 1000 == 0:
                print "1000 cycle time : {}".format(time.time()-tick)
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
        globalmean = self.global_movie_mean
        usermean = self.user_ratings[user]
        moviemean=self.movie_ratings[item]
        bxi = self.global_movie_mean + self.user_ratings[user] + self.movie_ratings[item]
        item_list, similarity_vector = self.select_nearest(item,user)
        sum_similarity = sum(similarity_vector)
        weighted_rating = 0
        # Use len of itemlist since it may contain fewer than expected elements
        # Depends on how many movies the user has rated
        #print len(item_list)
        for ix in xrange(len(item_list)):
            movie = item_list[ix]
            bxj = self.global_movie_mean + self.user_ratings[user] + self.movie_ratings[movie]
            rxj = self.training_matrix_csr[movie, user]
            weighted_rating = weighted_rating+ (rxj - bxj) * similarity_vector[ix]
        if sum_similarity == 0:
            return bxi
        else:
            pred_rating = weighted_rating/sum_similarity + bxi
        return pred_rating

    # Stub
    def collaborative_filter(self, filepath):
        self.cosine_similarity_matrix = self.load_sparse_matrix(filepath).tocsr()
        RMSE = self.collab_RMSE()
        #self.predict_rating(10, 10)
        print "Item-Item Collaborative Filtering RMSE: {}".format(RMSE)
        return None

        #Completely untested
    def select_nearest(self,reference_movie,user, k=5):
        similarity_vector = np.zeros((k,1))
        similarity_row = self.cosine_similarity_matrix.getrow(reference_movie)
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



    # if not os.path.isfile(centered_training_filepath):
    #     tick = time.time()
    #     recommender.center_matrix(recommender.training_matrix_coo,centered_training_filepath)
    #     print "Time to center matrix: {}".format(time.time()-tick)
    #     #recommender.test_center()