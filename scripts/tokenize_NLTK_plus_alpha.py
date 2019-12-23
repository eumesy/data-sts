import re
from nltk import word_tokenize
import nltk
nltk.download('punkt')

not_punc = re.compile('.*[A-Za-z0-9].*')

def preprocess(t):
    t = t.lower().strip("';.:()").strip('"')
    t = 'not' if t == "n't" else t
    return re.split(r'[-]', t)

def sent2words_nltk_usif(sentence):
    tokens = []

    for token in word_tokenize(sentence):
        if not_punc.match(token):
            tokens = tokens + preprocess(token)

    return tokens
