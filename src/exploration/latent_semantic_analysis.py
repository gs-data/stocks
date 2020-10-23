if __name__ == '__main__':
    import os
    import sys
    sys.path[0] = os.getcwd()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from src.data.raw_twitter_to_en_text import raw_twitter_to_english_text
from src.util.nlp_concepts import get_concepts_from_svd
from src.util.nlp_concepts import print_top_terms

"""
Latent semantic analysis (LSA) of documents using
  truncated singular value decomposition (SVD) of a matrix
  to map from high dimensional "term" space 
         to low dimensional "concept" space

Given a document-term (tf-idf) matrix X, SVD gives
  X = U * S * V^T
The singular values measure weight of "concept"
The matrix V maps between terms and "concepts"
  each row of V^T is a "concept" expressed in term coordinates

Working with sklearn API:
svd = TruncatedSVD(n_components=dimension_of_concept_space)

svd.fit_transform(X) returns X * V = U * S
  this is the representation of documents in the "concept" space

svd.components_ is equal to the matrix V^T
"""


path_to_raw_data = './data/raw/twitter_stream.txt'

# Create document term tfidf sparse matrix
tfidf = TfidfVectorizer(ngram_range=(1, 3),
                        strip_accents='unicode',
                        stop_words='english')
sparse = tfidf.fit_transform(raw_twitter_to_english_text(path_to_raw_data))
print("Document-Term matrix size:", sparse.shape)


# Choose dimension of the "concept" space
num_concepts = 500


# Compute dimensionality reduction
svd = TruncatedSVD(n_components=num_concepts)
reduced = svd.fit_transform(sparse)

# Measure the effectiveness of the reduction
print("Explained Variance Ratio:", sum(svd.explained_variance_ratio_))
# Note that the sum of squares of all singular values equals
#   the Frobenius norm of the matrix (ie entry-wise l2 norm).
#   The TfidfVectorizer normalizes all rows to have l2 norm of 1,
#   so the sum of squares of all singular values is the number of rows
print("Ratio of sum of squares of singular values:", sum(svd.singular_values_ ** 2) / sparse.shape[0])

# Matrix V^T from decomposition each row is a "concept"
concepts = get_concepts_from_svd(tfidf, svd)


# Choose number of concepts to view
num_of_top_concepts = 25
# Choose number of terms to view for each concept
num_of_top_terms = 12


# Print top terms from the top concepts
print_top_terms(concepts, num_of_top_terms, num_of_top_concepts)
