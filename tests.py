from Autocomplete.model import NgramCounter, NgramModel

ng_counter = NgramCounter('data/tokenized_emma', ngram_len=3)
counts = ng_counter.count()

ngram_model = NgramModel(ng_counter)
ngram_model.train(k=1)

print('text generation starts')
ngram_model.update_context(3)
res = ngram_model.generate_text(target_words='to be', number_of_sents=5)
print(res)
