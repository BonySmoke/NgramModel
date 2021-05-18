'''
Transform any given text to n-grams
'''
from nltk.corpus import gutenberg, stopwords
# from ngram_model import NgramCounter
import random

def word_ngrams(text, n=2, stop=False):
    '''
    Devide the text to word n-grams depending on the "n"
    params:
        text: str -> the actual text
        n: int -> the depth of n-grams
        stop: bool -> whether the stop words should be removed
    '''
    stop = set(stopwords.words("english")) if stop else {}

    token = [token for token in text.lower().split() if token != "" if token not in stop]

    z = zip(*[token[i:] for i in range(n)])
    ngrams = [' '.join(ngram) for ngram in z]

    return ngrams

filtered_corpus = list()
for book in gutenberg.fileids():
    ngrams_text = word_ngrams(gutenberg.raw('austen-emma.txt'))
    filtered_corpus.append(ngrams_text)

end_corpus = {text for text_arr in filtered_corpus for text in text_arr}


def text_generator(start_word, texts, num_of_chains=2):
    output = start_word

    for _ in range(num_of_chains):
        matches = [ngram for ngram in end_corpus if ngram.startswith(start_word)]
        if len(matches):
            match = random.choice(matches)
            res = match.split(' ')[-1]
            output += f' {res}'
            start_word = match.split(' ')[-1]

    return output

print(text_generator('tree', end_corpus, num_of_chains=5))
