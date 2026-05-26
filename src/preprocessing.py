"""
Text preprocessing utilities for the Cornell Movie Review sentiment analysis pipeline.

Provides tokenisation, stemming, and lemmatisation functions used by all downstream
feature extractors.
"""

import string
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Ensure required NLTK data is available
for _pkg in ("stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


def tokenizeText(text, parameter):
    """
    Tokenise and normalise a single text string.

    Removes stopwords and punctuation, applies a minimum 3-character filter,
    and optionally applies stemming or lemmatisation.

    Parameters
    ----------
    text : str
        Raw input text.
    parameter : str
        Normalisation mode.  Use ``"stemmer"`` for Porter stemming or
        ``"lemma"`` for WordNet lemmatisation.

    Returns
    -------
    list of str
        Processed token list.
    """
    stoplist = set(stopwords.words("english"))
    punctuation = string.punctuation

    tokens = word_tokenize(text)
    regex = RegexpTokenizer(r"\w+")
    stemmer = PorterStemmer()
    lemma = WordNetLemmatizer()

    words = [w.lower() for w in tokens]

    wordList = []
    for word in words:
        if word not in stoplist and word not in punctuation and word != "":
            if parameter == "lemma":
                word = lemma.lemmatize(word)
            elif parameter == "stemmer":
                word = regex.tokenize(word)
            word = "".join(word)
            if len(word) < 3:
                continue
            wordList.append(word)

    return wordList


def processTexts(texts):
    """
    Apply lemmatisation to a list of raw texts.

    Parameters
    ----------
    texts : list of str
        Raw text corpus.

    Returns
    -------
    list of list of str
        Each element is a list of lemmatised tokens for the corresponding text.
    """
    processedTexts = []
    for text in texts:
        processed = tokenizeText(text, "lemma")
        processedTexts.append(processed)
    return processedTexts


def stemmingTokenize(texts):
    """
    Apply Porter stemming to each text and return whitespace-joined strings.

    Parameters
    ----------
    texts : list of str
        Raw text corpus.

    Returns
    -------
    list of str
        Each element is a single whitespace-joined string of stemmed tokens,
        ready for use with scikit-learn vectorisers.
    """
    wordList = []
    for text in texts:
        words = tokenizeText(text, "stemmer")
        wordList.append(" ".join(words))
    return wordList


def lemmaTokenize(texts):
    """
    Apply WordNet lemmatisation to each text and return whitespace-joined strings.

    Parameters
    ----------
    texts : list of str
        Raw text corpus.

    Returns
    -------
    list of str
        Each element is a single whitespace-joined string of lemmatised tokens,
        ready for use with scikit-learn vectorisers.
    """
    wordList = []
    for text in texts:
        words = tokenizeText(text, "lemma")
        wordList.append(" ".join(words))
    return wordList
