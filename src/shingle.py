import numpy as np



class Shingling():
    """Class that implements the shingling functionality."""
    def __init__(self, k):
        """
        k: Size of the shingles.
        """
        self.k = k
        self.hash = hash # Use python built-in hashing function
    
    def _transform(self, doc):
        """
        Transform a document into a sorted set of shingles.
        """
        # Compute shingles
        shingles = np.array([doc[i:i+self.k] for i in range(0, len(doc) - self.k + 1)])
        # Filter out duplicates and sort
        hashes = sorted(set([self.hash(shingle) for shingle in shingles]))

        # Returned hashed shingles
        return hashes

    def transform(self, collection):
        """
        Transfrom a collection of documents into shingles.
        Returns a list of sets.
        """
        return [self._transform(doc) for doc in collection]


def compare_shingles(a, b):
    """Returns the jaccard similarity between two sets.
        a -> set
        b -> set
    """
    return len(set(a) & set(b)) / len(set(a) | set(b))