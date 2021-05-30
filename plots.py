import matplotlib.pyplot as plt
from Autocomplete.ngram_model import NgramCounter, NgramModel
from itertools import islice 

plot_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
hamlet = [-9.64, -12.58, -10.78, -8.88, -7.04, -5.36, -3.93, -2.68, -1.64, -0.82]
game_of_thrones = [-9.81, -13.53, -13.05, -11.77, -10.49, -9.29, -8.19, -7.21, -6.34, -5.57]

#---AVERAGE LOG PROBABILITY---
# plt.plot(plot_x, hamlet, label='hamlet')
# plt.plot(plot_x, game_of_thrones, label='game of thrones')
# plt.xlabel('N-gram length')
# plt.ylabel('Average Log likelihood')
# plt.title('Probability Comparison: Hamlet vs Game of Thrones')
# plt.legend()
# plt.show()

#---Ngram Counter---
ng_counter = NgramCounter('data/tokenized_emma', ngram_len=3)
counts = ng_counter.count()
# sorted_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
# dict_ = dict(islice(sorted_counts.items(), 10))
# plt.title('N-gram distribution')
# plt.bar(list(dict_.keys()), list(dict_.values()))
# plt.ylabel('count')
# plt.xlabel('N-gram')
# plt.show()

#---Ngram Model Probs---
ngram_model = NgramModel(ng_counter)
ngram_model.train(k=0.01)

sorted_probs = {k: v for k, v in sorted(ngram_model.probs.items(), key=lambda item: item[1], reverse=True)}
dict_ = dict(islice(sorted_probs.items(), 10))
plt.title('N-gram probability')
plt.bar(list(dict_.keys()), list(dict_.values()))
plt.ylabel('probability')
plt.xlabel('N-gram')
plt.show()