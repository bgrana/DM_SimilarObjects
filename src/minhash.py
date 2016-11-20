import random

import numpy as np


class MinHashing():
    """Class to compute minHash signatures."""
    def __init__(self, k):
        """
        k: length of the signatures. 
        """
        self.k = k
        self.c = 4294967311 # Big prime
        # Compute random coefficients
        self.a_coeffs = random.sample(range(1, 2**31), k)
        self.b_coeffs = random.sample(range(1, 2**31), k)
        
    def _hash(self, x, i):
        """
        Compute (ax+b)%c hash function.
        x:  input number to hash.
        i:  index to select coefficients.
        """
        return (self.a_coeffs[i]*x + self.b_coeffs[i])%self.c
        
    def _transform(self, shingles):
        """
        Transfrom a set of hashed shingles (vector of integers)
        into a signature (vector of integers).
        """
        signatures = np.array([[self._hash(shingle, i) for shingle in shingles] for i in range(self.k)])
        return np.array([l[np.argmin(l)] for l in signatures])

    def transform(self, collection):
        """
        Transform a collection of documents expressed as sets 
        of shingles into a matrix of signatures.
        Each row represents a component of the signature and
        each column a different document.
        """
        return np.array([self._transform(shingleset) for shingleset in collection]).T


def compare_signatures(a, b):
    """Computes the similarity between 2 signatures.
        a: numpy array
        b: numpy array
    """
    if len(a) != len(b):
        raise ValueError("Signatures lengths differ.")
    return np.mean(a == b)
