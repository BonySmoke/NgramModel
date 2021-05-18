from nltk.corpus import stopwords
from nltk import RegexpTokenizer, sent_tokenize

def get_tokenized_sentences(file: str):
    '''
    Tokenize one sentence at a time in a file
    :param file: str
        file with a text
    '''
    with open(file, 'r') as f:
        for sentence in f.read().splitlines():
            token_sentence = sentence.lower().split( )
            yield token_sentence

def generate_tokenized_sentences(paragraph: str):
    """
    Tokenize each sentence in paragraph.
    For each sentence, tokenize each words and return the tokenized sentence one at a time.
    :param paragraph: text of paragraph
    """
    word_tokenizer = RegexpTokenizer(r'[-\'\w]+')

    for sentence in sent_tokenize(paragraph):
        tokenized_sentence = word_tokenizer.tokenize(sentence)
        if tokenized_sentence:
            tokenized_sentence.append('[END]')
            yield tokenized_sentence

def replace_characters(text: str) -> str:
    """
    Replace punctuations that can mess up sentence tokenizers
    :param text: text with non-standard punctuations
    :return: text with standardized punctuations
    """
    replacement_rules = {'“': '"', '”': '"', '’': "'", '--': ','}
    for symbol, replacement in replacement_rules.items():
        text = text.replace(symbol, replacement)
    return text

def tokenize_raw_text(raw_text_path: str, token_text_path: str) -> None:
    """
    Read a input text file and write its content to an output text file in the form of tokenized sentences
    :param raw_text_path: path of raw input text file
    :param token_text_path: path of tokenized output text file
    :return: None
    """
    with open(raw_text_path, encoding='utf-8') as read_handle, open(token_text_path, 'w') as write_handle:
        for paragraph in read_handle:
            paragraph = replace_characters(paragraph.lower())

            for tokenized_text in generate_tokenized_sentences(paragraph):
                write_handle.write(' '.join(tokenized_text))
                write_handle.write('\n')

def word_ngrams(text, n=2, stop=False):
    '''
    Devide the text to word n-grams depending on the "n"
    params:
        text: str -> the actual text
        n: int -> the depth of n-grams
        stop: bool -> whether the stop words should be removed
    '''
    stop = set(stopwords.words("english")) if stop else {}

    token = [token for token in text if token != "" if token not in stop]

    z = zip(*[token[i:] for i in range(n)])
    ngrams = [' '.join(ngram) for ngram in z]

    yield ngrams
    