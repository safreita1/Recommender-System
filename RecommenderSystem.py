import time
import scipy.sparse as sparse

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

    default = [0.0002, 0.02]

    learning_rate_trials = [0.015, 0.006, 0.002]
    lambda_reg_trials = [0.06, 0.02]
    k_trials = [5, 10, 20]

    start_whole = time.time()

    recommender = RecommenderSystems(dataset=dataset)
    recommender.initialize_program()

    baseline = BaselineRecommendations(training_matrix_coo=recommender.training_matrix_coo, test_matrix_coo=recommender.test_matrix_coo)
    baseline.run_baseline()
    latent_model = LatentFactorModel(epochs=50, k=k_trials[2], learning_rate=learning_rate_trials[0], lambda_reg=lambda_reg_trials[0], training_coo=recommender.training_matrix_coo,
                                     test_coo=recommender.test_matrix_coo, user_average=baseline.user_average)

    # Overrides default parameters passed in constructor
    latent_model.run_new_model()
    #latent_model.run_old_model(model_directory='optimization/2017-11-23_16-28-35/')

    #recommender.fill_similarity_matrix(centered_training_filepath, similarity_filepath)
    #recommender.collaborative_filter(similarity_filepath)

    end = time.time()
    print "Time to run program: " + str((end - start_whole))
