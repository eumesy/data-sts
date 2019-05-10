#!/usr/bin/env python3

import argparse

import joblib
from tqdm import tqdm
import stanfordnlp

def sent2words(sentence, nlp):
    """
    args:
    - sentence (eg 'This is a sentence.')
    - nlp: StanfordNLP model

    return:
    - list of words
        eg ['This', 'is', 'a', 'sentence', '.']

    memo:
    - to lemmatize, `token.words[0].lemma`
    - w/lemmatize, 2x slower
    - w/parsing, 10x slower
    """

    doc = nlp(sentence)
    return [token.text
            for s in doc.sentences
                for token in s.tokens]

def line2wordspair(line, nlp):
    """
    args:
    - line: `sent1[TAB]sent2[\n]`
    - nlp: StanfordNLP model

    return:
    - tuple of " list of words"
        eg (['This', 'is', 'first', 'sentence', '.'], ['This', 'is', 'second', 'sentence', '.'])
    """
    sent1, sent2 = line.rstrip().split('\t')
    return (
        sent2words(sent1, nlp),
        sent2words(sent2, nlp)
    )

def main():
    nlp = stanfordnlp.Pipeline(processors='tokenize')

    with open(args.path_infile) as fi:
        joblib.dump([line2wordspair(line, nlp) for line in tqdm(fi)],
                    args.path_outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='...'
    )
    parser.add_argument('path_infile',
                        type=str,
                        help='path to STS2012-2017 formatted TSV file including sentence pairs. each line is of the form: `sent1[TAB]sent2`. eg /path/to/STS.input.hoge.txt')
    parser.add_argument('path_outfile',
                        type=str,
                        help='path to output (eg /path/to/STS.input.hoge.en_ewt.joblib)')

    args = parser.parse_args()
    main()
