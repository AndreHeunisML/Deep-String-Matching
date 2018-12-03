

import pandas as pd
import spacy
from pylab import *


def switch(word):
    if len(word) <= 3:
        return word

    word = list(word)
    a = np.random.randint(0, len(word))
    b = np.random.randint(0, len(word))
    word[a], word[b] = word[b], word[a]

    return "".join(word)


def clean_component(doc):
    """ Clean up text. Tokenize, lowercase, and remove punctuation and stopwords """
    # Remove punctuation, symbols (#) and stopwords
    stops = list(spacy.lang.en.stop_words.STOP_WORDS)
    doc = [tok for tok in doc if not tok.text in stops and not tok.is_punct]
    doc = [tok.lemma_.lower() for tok in doc]

    doc = ' '.join(doc)
    return doc


def post_process(s):
    # append random stop words to the start and end
    stops = list(spacy.lang.en.stop_words.STOP_WORDS)
    stop_word_count = len(stops)
    pp = stops[np.random.randint(0, stop_word_count)] + ' ' + s + ' ' + stops[np.random.randint(0, stop_word_count)] if len(s) > 20 else s

    # randomly switch 2 letters
    pp = switch(pp)

    # randomly remove some vowels
    vowels = ['a', 'e', 'i', 'o', 'u']
    pp = ''.join([c if c not in vowels else c if np.random.randint(0, 100) > 10 else "" for c in pp])

    # randomly remove some spaces
    pp = ''.join([c if c != ' ' else c if np.random.randint(0, 100) > 10 else "" for c in pp])

    return pp


def get_all_news():
    yahoo = pd.read_csv(
        "/Users/andreheunis/Downloads/doc_classification_datasets/yahoo_answers_csv/train.csv",
        header=None)
    yahoo.columns = ['0', '1', '2', '3']
    print(len(yahoo))

    series = pd.concat([yahoo['1'], yahoo['2'], yahoo['3']])
    series = series.dropna().values

    # Load the large English NLP model
    print("Loading lang model")
    nlp = spacy.load('en_core_web_lg', disable=['tagger', 'parser', 'ner', 'textcat'])

    nlp.add_pipe(clean_component, name='clean_component', first=False)

    # The text we want to examine
    text = series
    text = [t for t in text if len(t) > 12 and len(t) < 33]

    print("Running pipe")
    doc = nlp.pipe(text)

    processed_docs = [str(s) for s in doc]

    print("Running post process")
    processed2_docs = [post_process(s) for s in text]

    series1 = pd.Series(text + text)
    series2 = pd.Series(processed_docs + processed2_docs)

    output = pd.concat([series1, series2], axis=1)
    output.columns = ['anchor', 'match']

    return output


if __name__ == "__main__":
    get_all_news()