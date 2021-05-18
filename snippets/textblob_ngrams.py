from textblob import TextBlob
from collections import Counter

sentence = "This has never been easier to generate ngrams"

blob = TextBlob(sentence)
res = blob.ngrams(n=2)

def count_ngrams(ngrams:list)->dict:
    ngrams_count = dict()
    for ngram in res:
        ng = ' '.join(ngram)
        if ng not in ngrams_count:
            ngrams_count[ng] = 1
        else:
            ngrams_count[ng] += 1
    
    return ngrams_count

def estimate_probability(counts, k=1):
    probs = dict()
    for key, value in counts.items():
            prob_nom = value + k # laplace smoothing
            prob_denom = len(sentence) + k * len(counts)
            probs[key] = prob_nom / prob_denom
        
    return probs

counts = count_ngrams(res)

print(estimate_probability(counts))