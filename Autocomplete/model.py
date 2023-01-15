from .utils import get_tokenized_sentences, word_ngrams
from math import log2
import random


class NgramCounter:

    def __init__(self, file_path: str = 'data/tokenized_emma', ngram_len=2):
        '''
        :params file_path: file
            the path to the file
        :param ngram_len: int
            the count of ngrams per iteration
        '''
        self.file_path = file_path
        self.ngram_len = ngram_len
        self.counts = dict()  # num of ngrams
        self.tokens = 0  # num of words in the text
        self.ngrams = list()  # the list of all ngrams

    def count(self):
        '''
        Counts the number of n-grams in the given text
        '''
        self.sentence_count = int()
        self.tokens = int()

        if self.ngram_len == 1:
            for sentence in get_tokenized_sentences(self.file_path):
                self.sentence_count += 1
                self.tokens += len(sentence)
                for ngram in sentence:
                    self.counts[ngram] = self.counts.get(ngram, 0) + 1
                    self.ngrams.append(ngram)

            return self.counts

        for sentence in get_tokenized_sentences(self.file_path):
            self.sentence_count += 1
            self.tokens += len(sentence)
            for ngram in word_ngrams(text=sentence, n=self.ngram_len):
                for token in ngram:
                    self.counts[token] = self.counts.get(token, 0) + 1
                    self.ngrams.append(token)

        return self.counts


class NgramModel:

    def __init__(self, ngram_counter: NgramCounter):

        # text comparison
        self.probs = dict()
        self.ngram_counter = ngram_counter
        # the count of each word in a dictionary
        self.counts = ngram_counter.counts.copy()
        # the sign of the words that are not in a dictionary
        self.counts['[UNKNOWN]'] = 0
        # all unique words in a NgramCounter
        self.known_words = set(ngram_counter.counts.keys())
        # the num of words in the NgramCounter
        self.vocabulary_size = len(self.known_words)

        # text generation
        self.context = dict()  # dictionary that keeps list of candidate words given context
        self.ngrams = ngram_counter.ngrams

        self.update_context()

    def train(self, k):
        '''
        Count the probability of each dictionary word in a training corpus
        :param k: int
            Laplace smoothing. Used for unknown ngrams
        '''
        for key, value in self.counts.items():
            prob_nom = value + k  # probability numerator
            prob_denom = self.ngram_counter.tokens + k * \
                self.vocabulary_size  # probability denominator
            self.probs[key] = prob_nom / prob_denom

        return self.probs

    def estimate_word_probability(self, text_evaluation: NgramCounter):
        '''
        Calculate an average log likelihood for a text
        '''
        test_log_likelihood = int()
        self.unknown_unigram_sign = '[UNKNOWN]'
        self.text_evaluation = text_evaluation.counts

        for key, value in self.text_evaluation.items():
            if key not in self.known_words:
                key = self.unknown_unigram_sign  # if the word is not in the train model
            train_prob = self.probs[key]  # the word probability in train model
            log_likelihood = value * log2(train_prob)
            test_log_likelihood += log_likelihood

        avg_test_log_likelihood = test_log_likelihood / text_evaluation.tokens
        return avg_test_log_likelihood

    def update_context(self) -> list:
        '''
        Find all the possible ngram continuations
        e.g. "the day":["was", "has"...]

        :return dictionary: the vocabulary with the ngrams and their continuations
        '''
        ngrams = self.ngrams
        num_of_chains = self.ngram_counter.ngram_len

        for i in range(len(ngrams)-1):
            prev_words = ngrams[i]  # get the current ngram
            # next ngram without the words from prev_words
            next_word = ngrams[i+1].split()[num_of_chains-1:]

            if prev_words in self.context:
                self.context[prev_words].append(''.join(next_word))
            else:
                self.context[prev_words] = next_word

        return self.context

    def context_prob(self, context, token):
        '''
        Calculates probability of a candidate token to be generated given a context
        :return: conditional probability
        '''
        try:
            count_of_token = self.ngram_counter.counts[f'{context} {token}']
            count_of_context = float(len(self.context[context]))
            probability = count_of_token / count_of_context

        except KeyError:
            probability = 0.0

        return probability

    def random_word(self, context: list) -> str:
        '''
        Given the context, build the word sequence
        e.g. day way -> grey = day was grey
        :param context: the context word(s)
        :return:
        '''
        r = random.random()
        cont = ' '.join(context)
        tokens_probs = {}
        total = int()

        if cont not in self.context:
            raise KeyError(
                f"Context '{cont}' does not have a candidate word"
            )

        related_tokens = self.context[cont]

        if len(related_tokens) > 1 and '[end]' in related_tokens:
            related_tokens.remove('[end]')

        for token in related_tokens:
            tokens_probs[token] = self.context_prob(context, token)

        for token in sorted(tokens_probs):
            total += tokens_probs[token]
            if total > r:
                return token

        # random token if no token was returned
        return random.choice(related_tokens)

    def choose_random_ngram(self, context: str) -> str:
        '''
        Choose a random ngram containing the context word
        :param contaxt str: a random (sequence of) word(s)
        :return str: an ngram containing the context
        '''
        context = context.lower()
        try:
            if len(context.split()) > self.ngram_counter.ngram_len:
                error = f'The context word(s) should not exceed {self.ngram_counter.ngram_len}'
                raise Exception(error)

            ngram = random.choice(
                [n for n in self.ngram_counter.counts.keys()
                 if n.startswith(context)])

            return str(ngram)
        except Exception as e:
            print(e.args[0])
            return 'The n-gram containing the word(s) not found. Please provide another n-gram.'

    def generate_text(self, target_words: str, number_of_sents: int) -> str:
        '''
        Generate Text
        :param target_word: the beginning of the sentence
        :param number_of_chains: number of sequences to be generated
            with the given context
        :return: text
        '''
        n = self.ngram_counter.ngram_len  # len of ngram
        context = self.choose_random_ngram(target_words)
        context_sequence = context.split()

        sents = []

        sent = [context]

        # iterations are set to prevent creating text that loops over itself
        allowed_iter = 5000
        current_iter = 0
        while (len(sents) != number_of_sents
               and allowed_iter >= current_iter):
            # choose a random ngram from the target if sentence is empty
            if not sent:
                context_sequence = self.choose_random_ngram(target_words)
                context_sequence = context_sequence.split()

            word = self.random_word(context_sequence)
            sent.append(word)
            if n > 1:
                # drop the first word from the context
                context_sequence.pop(0)

                context_sequence.append(word)

            if '[end]' in word:
                complete: str = " ".join(sent)
                complete = f'{complete.partition("[end]")[0].strip()}.'.capitalize(
                )
                sents.append(complete)
                sent = []

            current_iter += 1

            if current_iter == allowed_iter:
                raise Exception(
                    f"Could not construct text in {allowed_iter} iterations")

        return ' '.join(sents)
