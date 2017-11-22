import math
import time
import itertools
import scipy.sparse as sparse
from collections import defaultdict

from LatentFactorModel import LatentFactorModel
from BaselineRecommendations import BaselineRecommendations



class RecommenderSystems:
    def __init__(self, dataset):
        self.training_matrix_coo = None
        self.test_matrix_coo = None
        self.training_filepath = 'matrices/{}_training.npz'.format(dataset)
        self.testing_filepath = 'matrices/{}_test.npz'.format(dataset)
        self.centered_training_filepath = 'matrices/{}_centered_training.npz'.format(dataset)
        self.centered_test_filepath = 'matrices/{}_centered_test.npz'.format(dataset)
        self.similarity_filepath = 'matrices/{}_similarity.npz'.format(dataset)

    def load_sparse_matrix(self, file_name):
        return sparse.load_npz(file_name)

    def initialize_program(self):
        # Load the sparse matrix from a file
        self.training_matrix_coo = self.load_sparse_matrix(self.training_filepath)
        self.test_matrix_coo = self.load_sparse_matrix(self.testing_filepath)


if __name__ == '__main__':
    # Uncomment only one dataset
    dataset = 'random'
    #dataset = 'arbitrary'

    start_whole = time.time()

    recommender = RecommenderSystems(dataset=dataset)
    recommender.initialize_program()

    baseline = BaselineRecommendations(training_matrix_coo=recommender.training_matrix_coo, test_matrix_coo=recommender.test_matrix_coo)
    baseline.run_baseline()
    latent_model = LatentFactorModel(epochs=50, k=5, learning_rate=0.0002, lambda_reg=0.02, training_coo=recommender.training_matrix_coo,
                                     test_coo=recommender.test_matrix_coo, user_average=baseline.user_average)
    latent_model.run_model()


    #recommender.fill_similarity_matrix(centered_training_filepath, similarity_filepath)
    #recommender.collaborative_filter(similarity_filepath)

    end = time.time()
    print "Time to run program: " + str((end - start_whole))
