import numpy as np
import math
from collections import Counter
from enum import Enum

class MyFeatureGenerator:

    def __init__(self, vocab):
        # set(('all',), ('n-grams',), ('found',), ...)
        self.vocab = vocab
        # [{term1: 0, ...}, {term1: 2, ...}, ...]
        self.term_counts = []
    

    def generate_features(self, reviews, normalisation_technique):
        # reviews is [[('this', 'is', 'a'), ('is', 'a', 'review'), ('a', 'review', None)]]
        self.__collect_terms(reviews)

        if (normalisation_technique == NormTech.TF_IDF):
            return self.__tf_idf()
        elif (normalisation_technique == NormTech.FREQ_NORM):
            return self.__freq_norm()
        elif (normalisation_technique == NormTech.PPMI):
            return self.__ppmi(reviews)
        elif (normalisation_technique == NormTech.ONE_HOT):
            return self.__one_hot()
    

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
        # vectorise the term counts (0.0 if not present)
        vectors = [[review.get(term, 0.0) for term in self.vocab] for review in self.term_counts]
        row_sums = np.sum(vectors, axis=1, keepdims=True)
        # freq normalise vectors along the rows
        vectors = vectors / row_sums
        
        return vectors
    

    def __ppmi(self, reviews):
        context_window = 3
        # index for each term to look up matrix with
        term_indices = {term: i for i, term in enumerate(self.vocab)}

        def get_context_slice(review, i):
            return review[max(0, i - context_window) : min(i + context_window+1, len(review))]

        def get_cooccurence_matrix():
            # how many times a term appears in the context of another term
            matrix = np.zeros((len(self.vocab), len(self.vocab)))

            for review in reviews:
                for i, term in enumerate(review):
                    # create a context window slice of review
                    context = get_context_slice(review, i)
                    # count the co-occurences in the context
                    for context_term in context:
                        if context_term == term: continue
                        matrix[term_indices[term]][term_indices[context_term]] += 1

            return matrix
        
        def get_ppmi_matrix(cooccurence_matrix):
            total = np.sum(cooccurence_matrix)
            # p(w,c) is matrix of (num of times two words occur in same context / total)
            p_w_c = cooccurence_matrix / total
            # p(w) is list of (sum of row / total)
            p_w = np.sum(cooccurence_matrix, axis=1, keepdims=True) / total
            p_w[p_w == 0.0] = 1e-12 # avoid division by 0
            # p(c) is list of (sum of col / total)
            p_c = np.sum(cooccurence_matrix, axis=0, keepdims=True) / total
            p_c[p_c == 0.0] = 1e-12

            matrix = np.log2(p_w_c / (p_w * p_c))
            # negative values must be 0
            matrix[matrix < 0] = 0

            return matrix
        
        cooccurence_matrix = get_cooccurence_matrix()
        ppmi_matrix = get_ppmi_matrix(cooccurence_matrix)

        # vectorise by averaging ppmi score in context window
        vectors = np.zeros((len(reviews), len(self.vocab)))
        for i, review in enumerate(reviews):
            for j, term in enumerate(review):
                term_index = term_indices[term]

                # get average of window
                context = get_context_slice(review, j)
                ppmi_average = 0.0
                for context_term in context:
                    if context_term == term: continue
                    ppmi_average += ppmi_matrix[term_index][term_indices[context_term]] / (2 * context_window)

                # add to existing average for that term in doc
                vectors[i][term_index] += ppmi_average
        
        return vectors

    
    def __one_hot(self):
        one_hots = [[1 if term in term_count else 0 for term in self.vocab] for term_count in self.term_counts]

        return one_hots


class NormTech(Enum):
    TF_IDF = 1
    FREQ_NORM = 2
    PPMI = 3
    ONE_HOT = 4
    