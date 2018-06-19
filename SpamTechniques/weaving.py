from Preprocess.preprocess import tokenize_sentence
from Preprocess.preprocess import retrieve_sentences
from random import randint,seed
import numpy as np
from copy import deepcopy

def create_new_document_by_weaving(text,query,threshold):
    seed(1)
    sentences = retrieve_sentences(text)
    query_tokens,query_words = tokenize_sentence(query)
    new_sentences=[]

    for sentence in sentences:
        tokens,words = tokenize_sentence(sentence)
        new_words = deepcopy(words)
        places = np.random.random(len(words))
        for i,place in enumerate(places):
            if place > threshold:
                index = randint(0, len(query_words) - 1)
                query_word = query_words[index]
                diff = abs(len(new_words)-len(words))
                new_words.insert(i+diff,query_word)
        new_sentences.append(" ".join(new_words))
    new_text = "\n".join(new_sentences)
    return new_text






