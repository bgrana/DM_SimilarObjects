from collections import Counter
from itertools import combinations
from math import log

import numpy as np
from numpy.linalg import norm
from scipy.special import lambertw


class LSH:
    """Locality-sensitive hashing implementation."""
    def __init__(self, matrix, t):
        """
        matrix: numpy matrix of signatures
        t:      threshold for considering two signatures as similar    
        """
        self.t = t
        self.M = matrix
        # Compute number of rows and bands with respect to the
        # provided threshold
        self.k = self.M.shape[0]
        self.r = int(np.real((-lambertw(-self.k*log(t))/log(t))))
        self.b = int(self.k/self.r)
        # Truncate number of bands if b*r != k (to englobe all rows)
        self.b = self.b+1 if self.b*self.r!=self.k else self.b
        # Select norm of the vector as the hashing function
        self.hash = norm

    def index(self):
        """
        Compute the hashes of the matrix of signatures 
        and save them into a hashtable.
        """
        # Initialize buckets
        self.hashtables = []
        # Compute hash value for each signature
        hashes = [[self.hash(self.M[i*self.r:(i+1)*self.r, j])
                    for j in range(self.M.shape[1])]
                        for i in range(0, self.b)]
        # Fill buckets
        for j,band in enumerate(hashes):
            self.hashtables += [{}]
            for i,h in enumerate(band):
                if h not in self.hashtables[j]:
                    self.hashtables[j][h] = []
                self.hashtables[j][h] += [i]

    def get_pairs(self):
        """
        Get pairs of probable similar signatures according
        to the provided threshold. Returns 2-tuples of indices
        according to the given matrix.
        """
        if not hasattr(self, 'hashtables'):
            raise Error("Call index first")
        pairs = []
        # Get combinations of pairs of elements
        # in each bucket
        for table in self.hashtables:
            for bucket in table.values():
                pairs += combinations(bucket, 2)
        # Count appearances and get only pairs that appear 
        # in the same bucket a fraction greater than the specified
        #threshold
        return [key for key,value in Counter(pairs).items()
                    if value >= self.t*self.b]
