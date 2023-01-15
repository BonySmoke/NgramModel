# Ngram Text Generator
A simple ngram model to generate text

# Introduction
The project is based on the incredible [work](https://medium.com/mti-technology/n-gram-language-models-70af02e742ad) by Khanh Nguyen.

This is a simple ngram model that can generate sequences of text (not logical) based on probability of word appearence in the given context.

# Usage
Below is the snippet of how the module can be used
```python
from Autocomplete.model import NgramCounter, NgramModel

ngram_len = 3  # the length of ngrams to form from the text

# Count the frequency of each ngram in the given file
ng_counter = NgramCounter(file_path='data/tokenized_emma',
                          ngram_len=ngram_len)
counts = ng_counter.count()

# Instantiate the ngram model with the pretrained counter
ngram_model = NgramModel(ng_counter)
text = ngram_model.generate_text(target_words='It is time', number_of_sents=1)
```

Result
```
It is time for us to part with them hating change of every kind that emma could not regret her having gone to miss bates with a most open eagerness never for the twentieth part of a moment it would have been impossible for any woman of sense to endure she spoke her resentment in a form of words perfectly intelligible to me in the whole of the profits of his mercantile life appeared so very moderate it was not more productive than such meetings usually are.
```

Ngram counter can be trained on one of the following books:
- Game of Thrones (tokenized_game_of_thrones)
- Emma (tokenized_emma)
- Hamlet (tokenized_hamlet)
