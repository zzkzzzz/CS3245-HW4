#!/usr/bin/python3
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
# from textblob import TextBlob
nltk.download('wordnet')

# Correct query using textblob
# def correct_query(query):
#     corrected_query = []
#     for word in query.split():
#         corrected_word = TextBlob(word).correct()
#         corrected_query.append(str(corrected_word))
#     return ' '.join(corrected_query)

# Correct query using
def correct_query(query):
    corrected_query = []
    for word in query.split():
        # Check if the word is in WordNet, if not, assume it's misspelled
        if not wordnet.synsets(word):
            # Get a list of similar words based on edit distance
            similar_words = sorted(wordnet.all_lemma_names(), 
                                   key=lambda w: nltk.edit_distance(word, w))[:3]
            # Use the most similar word as the correction
            corrected_word = similar_words[0] if similar_words else word
        else:
            corrected_word = word
        corrected_query.append(corrected_word)
    return ' '.join(corrected_query)


def expand_query(query, num_words=2):
    expanded_query = []
    for word in query.split():
        synsets = wordnet.synsets(word)
        if len(synsets) == 0:
            return query
        top_words = get_top_words(synsets, num_words)
        expanded_query.extend(top_words)
        
    return ' '.join(expanded_query)


# helper function to get top words from synsets
def get_top_words(synsets, num_words=3):
    all_words = []
    for synset in synsets:
        lemmas = synset.lemma_names()
        all_words.extend(lemmas)
    word_counts = {}
    for word in all_words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, count in sorted_words][:num_words]
    
    return top_words



   
        
