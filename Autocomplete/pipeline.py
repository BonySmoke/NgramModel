from nltk import RegexpTokenizer, word_tokenize, sent_tokenize, FreqDist, bigrams
from itertools import islice
import utils

class DataParser:
    '''
    - Process data
    1) Parse and dump data.
    2) Load data.
    '''

    def __init__(self):
        pass

    def dump(self):
        pass

    def load(self):
        pass

    @staticmethod
    def tokenize_web_page(page_url: str, token_text_path: str) -> None:
        pass

    def ngram_counter(self, text_path, ngram_count=1) -> list:
        """
        Counts the number of n-grams in the given text
        :params:
        text_path: str
            the path to the file with n-grams
        ngram_count: int
            the n-gram sequence in the text
        :return:
        count of n-grams
        """
        self.sentence_count = int()
        self.counts = dict()
        self.tokens = int()

        for sentence in self.get_tokenized_sentences(text_path):
            self.sentence_count += 1
            self.tokens = len(sentence)
            for ngram in sentence:
                self.counts[ngram] = self.counts.get(ngram, 0) + 1

        return self.counts


utils.tokenize_raw_text('../data/emma.txt', '../data/tokenized_emma')
# ng_counter = DataParser()
# counts = ng_counter.ngram_counter(text_path='tokenized_emma')
# print(dict(islice(counts.items(), 3)))