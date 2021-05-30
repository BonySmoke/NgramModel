from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus) # create a matrix
print(vectorizer.get_feature_names()) # get ngrams
print(X.toarray())

ngram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))

X2 = ngram_vectorizer.fit_transform(corpus) # create a matrix
print(ngram_vectorizer.get_feature_names()) # get ngrams
print(X2.toarray())