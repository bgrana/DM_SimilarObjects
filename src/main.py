import argparse
import re
from operator import itemgetter
from os import listdir
from os.path import isfile, join

from lsh import LSH
from minhash import MinHashing, compare_signatures
from shingle import Shingling, compare_shingles


def main(args):
    # Get input params
    input_dir = args["dir"]
    th = args["th"]

    # Read all files contained in the input directory
    print("Loading documents...")
    onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    docs = []
    for fname in onlyfiles:
        with open(join(input_dir, fname), "r") as file:
            docs += [file.read()]

    # Clean documents removing trailing and duplicate blanks
    print("Cleaning documents...")
    docs = [re.sub('\W+', ' ', doc) for doc in docs]

    # Compute shingles of size n
    print("Computing shingles...")
    sh = Shingling(args["n"])
    shingles = sh.transform(docs)

    # Compute jaccard similarities
    print("Jaccard similarities (on hashed shingles) > " + str(th) + ":")
    similarities = {(onlyfiles[i],onlyfiles[j]): compare_shingles(shingles[i], shingles[j]) 
                    for i in range(0, len(docs)) 
                    for j in range(i+1, len(docs))}
    # Show similarities greater than the threshold
    print(sorted([(k,v) for k,v in similarities.items() 
                        if v > th], key=itemgetter(1), reverse=True))

    # Compute minHash signatures
    print("Computing signatures...")
    mh = MinHashing(args["k"])
    signatures = mh.transform(shingles)

    # Compute similarity esrimations
    print("Similarity estimations using minHashing > " + str(th) +":")
    estimations = {(onlyfiles[i],onlyfiles[j]):compare_signatures(signatures[:,i], signatures[:,j]) 
                for i in range(0, len(docs)) 
                for j in range(i+1, len(docs))}
    # Show similarity estimations greater than a threshold
    print(sorted([(k,v) for k,v in estimations.items() 
                        if v > th], key=itemgetter(1), reverse=True))

    # Show Differences between estimations and real similarities
    errors = {(onlyfiles[i],onlyfiles[j]):abs(estimations[(onlyfiles[i],onlyfiles[j])] - similarities[(onlyfiles[i],onlyfiles[j])])
              for i in range(0, len(docs)) 
              for j in range(i+1, len(docs))}
    # Show errors greater than 5%
    print("Estimaions with error greater than 5%:")
    print(sorted([(k,v) for k,v in errors.items()
                        if v > 0.05], key=itemgetter(1), reverse=True))

    # Apply LSH to find pairs of probable similar items
    lsh = LSH(signatures, th)
    lsh.index()
    candidates = lsh.get_pairs()

    # Show candidates
    print("Identified candidates with LSH:")
    print([(onlyfiles[t[0]],onlyfiles[t[1]]) for t in candidates])


if  __name__ =='__main__':
    # Parse input
    parser = argparse.ArgumentParser(description='Find similar documents using shingling, minhashing and LSH.')
    parser.add_argument('--dir', required=True, type=str, help='Input directory containing documents to compare.')
    parser.add_argument('--n', default=9, type=int, help='Size of the shingles to compute. Defaults to 9.')
    parser.add_argument('--k', default=100, type=int, help='Size of the signatures (i.e. number of hash functions to apply). Defaults to 100.')
    parser.add_argument('--th', default=0.8, type=float, help='Threshold of confidence for similarity. Defaults to 0.8.')

    args = vars(parser.parse_args())
    print(args)
    main(args)
