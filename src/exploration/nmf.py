if __name__ == '__main__':
    import os
    import sys
    sys.path[0] = os.getcwd()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from src.data.raw_twitter_to_en_text import raw_twitter_to_english_text
from src.util.nlp_concepts import get_concepts_from_nmf
from src.util.nlp_concepts import print_top_terms


path_to_raw_data = './data/raw/twitter_stream.txt'

# Create document term tfidf sparse matrix
tfidf = TfidfVectorizer(ngram_range=(1, 1),
                        strip_accents='unicode',
                        stop_words='english')
sparse = tfidf.fit_transform(raw_twitter_to_english_text(path_to_raw_data))
print("Document-Term matrix size:", sparse.shape)


# Choose dimension of the "concept" space
dim_of_concepts = 50


# Compute dimensionality reduction
nmf = NMF(n_components=dim_of_concepts)
reduced = nmf.fit_transform(sparse)

# Check convergence
print("Iterations to convergence:", nmf.n_iter_)
if (nmf.get_params()['max_iter'] <= nmf.n_iter_):
    print("CONVERGENCE WARNING: Iterations maxed out!")

# Matrix from factorization each row is a "concept"
concepts = get_concepts_from_nmf(tfidf, nmf)


# Choose number of concepts to view
num_of_top_concepts = 25
# Choose number of terms to view for each concept
num_of_top_terms = 12


# Print top terms from the top concepts
print_top_terms(concepts, num_of_top_terms, num_of_top_concepts)


