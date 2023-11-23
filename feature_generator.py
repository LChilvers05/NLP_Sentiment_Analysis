import numpy as np
import math
from collections import Counter
from enum import Enum

# TODO: normalise vectors
# TODO: filter out bad features?

class MyFeatureGenerator:

    # set(('all',), ('n-grams',), ('found',), ...)
    vocab = set()
    # [{term1: 0, ...}, {term1: 2, ...}, ...]
    term_counts = []

    def __init__(self, vocab):
        self.vocab = vocab
    
    def generate_features(self, reviews, normalisation_technique):
        # reviews is [[('this', 'is', 'a'), ('is', 'a', 'review'), ('a', 'review', None)]]
        self.__collect_terms(reviews)

        if (normalisation_technique == NormTech.TF_IDF):
            features = self.__tf_idf()
        elif (normalisation_technique == NormTech.FREQ_NORM):
            features = self.__freq_norm()
        elif (normalisation_technique == NormTech.PPMI):
            features = self.__ppmi()
            
        return features
    

    def __collect_terms(self, reviews):
        self.term_counts = []
        for review in reviews:
            # get the count of every term in each review
            self.term_counts.append(Counter(review))

        # TODO: create a count graph of features

    
    def __tf_idf(self):

        def __get_tf(term, review):
            # tf = number of occurences of the term in the review / number of terms in the review
            return review.get(term, 0)/sum(review.values())
        
        def __get_idf(review_count, review_with_term_count):
            # idf = log(number of reviews / number of reviews that contain the term)
            return math.log(float(review_count)/float(review_with_term_count + 1), 10)

        # is [(term1)[tf1, tf2, ...], (term2)[tf1, tf2, ...], ...]
        tf = np.zeros((len(self.vocab), len(self.term_counts)))
        # is [(term1)[idf1], (term2)[idf2], ...]
        idf = np.zeros((len(self.vocab), 1))

        for i, term in enumerate(self.vocab):
            review_with_term_count = 0
            for j, review in enumerate(self.term_counts):
                if term in review.keys(): review_with_term_count += 1
                # build tf matrix
                tf[i][j] = __get_tf(term, review)
            # build idf matrix
            idf[i][0] = __get_idf(len(self.term_counts), review_with_term_count)

        # tf-idf = tf * idf
        vectors = np.transpose(idf * tf)

        return vectors
    
    
    def __freq_norm(self):
        return
    
    
    def __ppmi(self):
        return
    
   
    
class NormTech(Enum):
    TF_IDF = 1
    FREQ_NORM = 2
    PPMI = 3