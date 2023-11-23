import numpy as np

class MyNaiveBayesClassifier:
    
    prior_probs = [] # [p(pos), p(neg), p(neu)]
    feature_probs = {} # {pos: {term1: ..., ...}, neg: {term1: ..., ...}, ...}

    def __set_prior_probs(self, labels):
        # movie review labels is [0(pos), 1(neg), 2(neu), 0, 2, 0, ...]
        # array of p(class) where index is class
        prior_probs = np.zeros(max(labels) + 1)
        self.class_count = len(prior_probs)
        for label in labels:
            prior_probs[label] += 1/len(labels) #TODO: log(1/len(labels))

        self.prior_probs = prior_probs
    
    #TODO:
    def __set_feature_probs(self, features, labels):

        for c in range(len(self.prior_probs)):
            self.feature_probs[c] = {}

            # get all features of a class [[pos review vec], [pos review vec], ...]
            class_features = features[labels == type]
            #p(lottery | spam) = num(spam emails with 'lottery' in training set) / num (spam emails in training set)
            for i, feature in features:
                self.feature_probs[c][feature] = np.sum(class_features[:, i]) / len(class_features)
        
        return
    
    def fit(self, x_data, y_data):
        self.__set_prior_probs(y_data)
        self.__set_feature_probs(x_data, y_data)
        
    
    