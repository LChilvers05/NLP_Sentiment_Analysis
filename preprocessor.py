import nltk
from nltk.corpus import stopwords

class MyPreprocessor:

    def __init__(self):
        self.vocab = set()
        

    def preprocess(self, reviews, 
                         is_lemmatisation = True, 
                         is_lowercase = True, 
                         is_stopwords_removed = True,
                         is_punctuation_removed = True, 
                         n_gram_len = 1
                         ):
        # tokenise each review: ['this', 'is', 'a', 'review']
        reviews = self.__tokenise(reviews, is_punctuation_removed, is_stopwords_removed, is_lowercase)

        # do lemmatisation or stemming: ['this', 'is', 'a', 'review']
        reviews = self.__lemmatise(reviews) if is_lemmatisation else self.__stem(reviews)

        # remove 'br' from ['a', 'review', ...]
        reviews = self.__clean(reviews)

        # generate n_grams: [[('this', 'is', 'a'), ('is', 'a', 'review'), ('a', 'review', None)], [...]]
        reviews = self.__n_gram(reviews, n_gram_len)

        self.__add_to_vocabulary(reviews)

        return reviews
    
    
    def __add_to_vocabulary(self, reviews):
        # get all the terms across all the reviews
        for review in reviews:
            for term in review: self.vocab.add(term)


    def __tokenise(self, reviews, is_punctuation_removed, is_stopwords_removed, is_lowercase):
        tokenised_reviews = []
        # tokenise each review using nltk
        remove_punc_tokeniser = nltk.RegexpTokenizer('\w+')
        stop_words = set(stopwords.words('english'))

        for review in reviews:
            # generate tokens with/out punctuation 
            tokens = remove_punc_tokeniser.tokenize(review) if (is_punctuation_removed) else nltk.word_tokenize(review)
            tokenised_review = []
            for token in tokens:
                # ignoring stopwords
                if (is_stopwords_removed and token.lower() in stop_words):
                        continue
                
                # converting to lowercase
                result_token = token.lower() if (is_lowercase) else token
                
                tokenised_review.append(result_token)
            tokenised_reviews.append(tokenised_review)
        
        return tokenised_reviews
    
                        
    def __lemmatise(self, reviews):
        lemmatised_reviews = []
        # do lemmatisation on tokenised reviews
        lemmatiser = nltk.WordNetLemmatizer()
        for review in reviews:
            # review is ['a', 'list', 'of', 'words']
            lemmatised_reviews.append([lemmatiser.lemmatize(token) for token in review])
        
        return lemmatised_reviews
    
    
    def __stem(self, reviews):
        stemmed_reviews = []
        # do stemming on tokenised reviews
        stemmer = nltk.PorterStemmer()
        for review in reviews:
            # review is ['a', 'list', 'of', 'words']
            stemmed_reviews.append([stemmer.stem(token) for token in review])
        
        return stemmed_reviews
    
    
    def __n_gram(self, reviews, n):
        # default to 1 if not valid length
        if (n < 1): n = 1
        # generate n-gram on processed reviews where
        # a review is ['a', 'list', 'of', 'words']
        return [list(nltk.ngrams(review, n)) for review in reviews]
    
    def __clean(self, reviews):
        return [[token for token in review if token != 'br'] for review in reviews]
    