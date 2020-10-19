import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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


# Load english tweet text from raw data
def project_raw_twitter_to_english_text(path_to_raw_data_file):
    """
    Yield english text from raw tweets
    """
    with open(path_to_raw_data_file, 'r') as raw:
        for line in raw:
            json_line = json.loads(line)
            if json_line['data']['lang'] == 'en':
                yield json_line['data']['text'].replace('\n', '\\n')


path_to_raw_data = './data/raw/twitter_stream.txt'

# Create document term tfidf sparse matrix
tfidf = TfidfVectorizer(ngram_range=(1, 1),
                        strip_accents='unicode')
sparse = tfidf.fit_transform(project_raw_twitter_to_english_text(path_to_raw_data))
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
concepts = pd.DataFrame(svd.components_, columns=tfidf.get_feature_names())

# Print top terms from the top concepts
num_of_top_terms = 10
num_of_top_concepts = 50
print('\n' + '\n'.join(
    [str(row) + ' ' + ',  '.join(
        [term for term in concepts.iloc[row].nlargest(num_of_top_terms).index])
     for row in range(num_of_top_concepts)])
)
