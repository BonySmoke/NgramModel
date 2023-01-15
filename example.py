from Autocomplete.model import NgramCounter, NgramModel

ngram_len = 3  # the length of ngrams to form from the text

# Count the frequency of each ngram in the given file
ng_counter = NgramCounter(file_path='data/tokenized_emma',
                          ngram_len=ngram_len)
counts = ng_counter.count()

# Instantiate the ngram model with the pretrained counter
ngram_model = NgramModel(ng_counter)
text = ngram_model.generate_text(target_words='It is time', number_of_sents=3)
print(text)
