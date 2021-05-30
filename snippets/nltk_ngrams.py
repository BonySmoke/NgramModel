from nltk.corpus import stopwords
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten, padded_everygram_pipeline
from nltk.lm import MLE

def generate_ngrams(text):

    stop = set(stopwords.words("english"))

    words = [token for token in text.lower().split() if token != "" if token not in stop]

    token = list(pad_sequence(words[:-1], n=2, pad_left=True, left_pad_symbol="<s>",
                pad_right=True, right_pad_symbol="</s>"))

    return list(ngrams(token, n=3))


text = "The sun is shining. It's very bright outside. Let us see if we can be better than those who betrayed us."

padded_sents = generate_ngrams(text)
train_data, vocab = padded_everygram_pipeline(3, padded_sents)

model = MLE(3)
model.fit(train_data, vocab)
print(model.counts) # the possible number of ngrams
#<NgramCounter with 3 ngram orders and 162 ngrams>
print(model.counts['sun']) # the number of ngrams with a word
#2
print(model.score('sun')) # the probability of a word
#0.031746031746031744
print(model.generate(20, random_seed=10)) # generate random sents
#['shining.', 'bright', 'outside.']