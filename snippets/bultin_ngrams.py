from nltk.corpus import stopwords

'''
My own ngrams implementation
'''

def generate_ngrams(text, n_gram=2, stop=True):
    '''
    N-gram generator
    '''
    stop = set(stopwords.words("english")) if stop else {}

    token = [token for token in text.lower().split() if token != "" if token not in stop]
    z = zip(*[token[i:] for i in range(n_gram)])
    ngrams = [" ".join(ngram) for ngram in z]

    return ngrams

text = "The sun is shining. Let's see it together. Maybe, there will be another days when we can see ourselves."
print(generate_ngrams(text=text))
