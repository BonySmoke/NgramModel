from Autocomplete.ngram_model import NgramCounter, NgramModel
from itertools import islice

#--------------
#Tests
ng_counter = NgramCounter('data/tokenized_emma', ngram_len=3)
counts = ng_counter.count()

ngram_model = NgramModel(ng_counter)
ngram_model.train(k=1)

print('text generation starts')
ngram_model.update_context(3)
random_ngram = ngram_model.choose_random_ngram(context_word='dress')
res = ngram_model.generate_text(random_ngram[0], 20)
print(res)

# --------------------------
# emma = NgramCounter('data/tokenized_emma', ngram_len=10)
# emma_counts = emma.count()
# sorted_emma_counts = {k: v for k, v in sorted(emma_counts.items(), key=lambda item: item[1], reverse=True)}
# print(dict(islice(sorted_emma_counts.items(), 10)))
# result = ngram_model.estimate_word_probability(emma)
# print(result)

# hamlet = NgramCounter('data/tokenized_hamlet', ngram_len=10)
# hamlet_counts = hamlet.count()
# sorted_hamlet_counts = {k: v for k, v in sorted(hamlet_counts.items(), key=lambda item: item[1], reverse=True)}
# print(dict(islice(sorted_hamlet_counts.items(), 10)))
# result2 = ngram_model.estimate_word_probability(hamlet)
# print(result2)

# source = NgramCounter('data/tokenized_data', ngram_len=2)
# source_counts = source.count()
# sorted_source_counts = {k: v for k, v in sorted(source_counts.items(), key=lambda item: item[1], reverse=True)}
# print(dict(islice(sorted_source_counts.items(), 10)))
# result3 = ngram_model.estimate_word_probability(source)
# print(result3)