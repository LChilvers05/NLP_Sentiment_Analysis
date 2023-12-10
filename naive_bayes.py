import numpy as np

class MyNaiveBayesClassifier:

    def __init__(self):
        self.prior_probs = [] # [p(pos), p(neg)]
        self.feature_probs = []
        

    def fit(self, x_data, y_data):
        self.prior_probs = self.__set_prior_probs(y_data)
        self.feature_probs = self.__set_feature_probs(x_data, y_data)


    def predict(self, x_data):
        class_count = self.__get_class_count()
        
        # is [(review1) argmax(log(p(class)) + sum(log(p(feature|class)))), (review2) ...]
        predictions = []

        for vector in x_data:
            prediction = self.prior_probs.copy()

            for i, feature in enumerate(vector):
                if feature == 0.0: continue
                for j in range(class_count):
                    prediction[j] += self.feature_probs[j][i]
            
            predictions.append(np.argmax(prediction))
        
        return predictions
    

    def __set_prior_probs(self, labels):
        # movie review labels is [0(pos), 1(neg), 2(neu), 0, 2, 0, ...]
        # array of p(class) where index is class
        prior_probs = np.zeros(max(labels) + 1)
        for label in labels:
            prior_probs[label] += 1/len(labels)
        prior_probs = list(map(np.log, prior_probs))

        return prior_probs
    
    
    def __set_feature_probs(self, vectors, labels, alpha=1.0):
        class_count = self.__get_class_count(labels)
        feature_count = vectors.shape[1]
        
        feature_probs = np.zeros((class_count, feature_count))

        # feature_probs is [(pos)[sum1, sum2, ...], (neg)[sum1, sum2, ...], ...]
        for i, vector in enumerate(vectors):
            for j, feature in enumerate(vector):
                feature_probs[labels[i]][j] += feature

        # feature_probs is [
        # (pos)[log(p(f1 | pos)), log(p(f2 | pos)), ...], 
        # (neg)[log(p(f1 | neg)), log(p(f2 | neg)), ...]
        # ]
        for i, class_features in enumerate(feature_probs):
            class_sum = np.sum(class_features)
            # alpha for laplace smoothing
            d = class_sum + (feature_count * alpha) 
            for j, class_feature in enumerate(class_features):
                n = class_feature + alpha
                feature_probs[i][j] = np.log(n/d)

        return feature_probs
    
    
    def __get_class_count(self, labels=[]):
        if len(self.prior_probs) == 0:
            self.__set_prior_probs(labels)
        return len(self.prior_probs)
      