
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import numpy as np


def normalization(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"([.!?,])", r" ", sentence)
    return sentence


def tokenization(sentence):
    return word_tokenize(sentence)


def stemming(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)


def bag_of_words(tokenized, word_list): # finding the words which are similer to each other
    tokenized = [stemming(words) for words in tokenized]
    bag = np.zeros(len(word_list), dtype=np.float32)
    for index, word in enumerate(word_list):
        if word in tokenized:
            bag[index] = 1.0
    return bag
# sentence=["hello","how","are","you"]
# words=["hi","hello","I","you","bye","thank","cool"]
# bag=bag_of_words(sentence,words)
# print(bag)
