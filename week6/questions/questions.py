import math
import os
import string
import sys
from collections import Counter

import nltk

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # dict to map filenames to file's contents
    corpus = dict()
    # iterating over directory
    for root, dirs, files in os.walk(directory):
        # getting filenames
        for name in files:
            # open each file
            with open(os.path.join(root, name), 'r') as f:
                # mapping content
                corpus[name] = f.read()
    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # tokenizing document
    document = nltk.tokenize.word_tokenize(document.lower())
    # list of words to remove
    remove_list = []

    for word in document:
        # listing unwanted words
        if word in string.punctuation \
            or word in nltk.corpus.stopwords.words("english") \
                or not word.isalpha():
            remove_list.append(word)

    # returning relevant words
    return [word for word in document if word not in remove_list]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # getting all words in documents
    words = set()
    for filename in documents:
        words.update(documents[filename])

    # calculating IDFs
    idfs = dict()
    for word in words:
        # number of documents containing word
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_rank = dict()
    for filename, filewords in files.items():
        file_tfidf = 0
        for word in query:
            if word in filewords:
                # counting word occurencies in file
                tf = Counter(filewords)[word]
                # calculating tf-idf
                file_tfidf += tf * idfs[word]
        file_rank[filename] = file_tfidf

    # sorting files based on td-idf values
    top_files = list(
        sorted(file_rank.items(), key=lambda item: item[1], reverse=True))
    # filtering filenames and n top files
    top_files = [data[0] for data in top_files][:n]
    
    return top_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_rank = dict()
    for sentence, s_words in sentences.items():
        sentence_idf = 0
        # query term density
        density = 0
        for word in query:
            if word in s_words:
                # calculating idf
                sentence_idf += idfs[word]
                density += 1
        density = density / len(s_words)
        sentence_rank[sentence] = sentence_idf, density

    # sorting sentences based on idf values and density
    top_sentences = list(
        sorted(sentence_rank.items(), key=lambda idf: idf[1], reverse=True))

    # filtering sentences and n top sentences
    top_sentences = [data[0] for data in top_sentences][:n]
    
    return top_sentences


if __name__ == "__main__":
    main()
