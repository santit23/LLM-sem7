import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from tfidf_utils import vocab, word_embeddings

pca = PCA(n_components=2)
reduced = pca.fit_transform(word_embeddings)

plt.figure(figsize=(10, 7))
for i, word in enumerate(vocab):
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word, fontsize=10)

plt.title("TF-IDF Word Embeddings (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.savefig("tfidf_word_embeddings.png")