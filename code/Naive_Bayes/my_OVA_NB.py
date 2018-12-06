import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse

count = 0

class myOVANB(MultinomialNB):

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)

        global count
        count = count + 1
        print('>>> ' + str(count) + ' Using my modified version of MNB implementing one versus all but one equation')

    def _count(self, X, Y):
        """Count feature occurrences."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        MN_numerator = self.feature_count_ + alpha
        MN_denuminator = self.feature_count_.sum(axis=1) + alpha * self.feature_count_.shape[1]
        self.MN_feature_log_prob_ = (np.log(MN_numerator) - np.log(MN_denuminator.reshape(-1, 1)))

        CM_numerator = self.feature_all_ + alpha - self.feature_count_
        CM_denuminator = self.feature_all_ - self.feature_count_
        CM_denuminator = CM_denuminator.sum(axis=1, keepdims=True) + alpha * self.feature_count_.shape[1]
        self.CM__feature_log_prob_ = (np.log(CM_numerator) - np.log(CM_denuminator.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')

        MN_likelyhood = safe_sparse_dot(X, self.MN_feature_log_prob_.T)
        CM_likelyhood = safe_sparse_dot(X, self.CM__feature_log_prob_.T)

        posterior_prob = self.class_log_prior_ + (MN_likelyhood - CM_likelyhood)

        return posterior_prob