import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot


class myMultinomialNB(MultinomialNB):

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        print('>>> Using my modified version of MN NB classifier')
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        numerator = self.feature_count_ + alpha
        denuminator = self.feature_count_.sum(axis=1) + alpha * self.feature_count_.shape[1]

        self.feature_log_prob_ = (np.log(numerator) -
                                  np.log(denuminator.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')

        posterior_prob = safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_

        return posterior_prob
