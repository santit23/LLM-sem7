from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
corpus = [
    "The cat sat on the mat.",
    "Dogs are loyal animals.",
    "Cats and dogs are popular pets.",
    "The dog barked at the stranger.",
    "The mat was clean and soft."
]

# Create the vectorizer
vectorizer = TfidfVectorizer()
# Fit and transform the corpus
x = vectorizer.fit_transform(corpus)
# Get the feature names
vocab = vectorizer.get_feature_names_out()
word_embeddings = x.T.toarray()
word_to_index = {word: i for i, word in enumerate(vocab)}

def get_embedding(word):
    word = word.lower()
    if word in word_to_index:
        return word_embeddings[word_to_index[word]]
    return None

def get_nearest_neighbor(word, k=3):
    embedding = get_embedding(word)
    similarities = cosine_similarity(embedding.reshape(1, -1), word_embeddings)[0]
    indices = similarities.argsort()[::-1]
    similar_words = [
        (vocab[i], similarities[i])
        for i in indices if vocab[i] != word][:k]
    return similar_words