if __name__ == '__main__':
    import os
    import sys
    sys.path[0] = os.getcwd()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import pandas as pd

from src.data.raw_twitter_to_en_text import raw_twitter_to_english_text
from src.util.nlp_concepts import get_concepts_from_svd
from src.util.nlp_concepts import get_concepts_from_svd_kmeans
from src.util.nlp_concepts import print_top_terms
from src.util.nlp_concepts import print_samples_of_text_by_label


path_to_raw_data = './data/raw/twitter_stream.txt'

# Create document term tfidf sparse matrix
tfidf = TfidfVectorizer(ngram_range=(1,3),
                        strip_accents='unicode',
                        stop_words='english')
sparse = tfidf.fit_transform(raw_twitter_to_english_text(path_to_raw_data))
print("Document-Term matrix size:", sparse.shape)


# Choose dimension of concept space
dim_of_concepts = 250
# Choose number of clusters
num_clusters = 100


# Setup pipeline
svd = TruncatedSVD(n_components=dim_of_concepts)
kmeans = KMeans(n_clusters=num_clusters)
pipeline = make_pipeline(svd, kmeans)

# Compute SVD and clusters
pipeline.fit(sparse)
if (kmeans.get_params()['max_iter'] <= kmeans.n_iter_):
    print("CONVERGENCE WARNING: Iterations maxed out!")

# Get cluster centers as possible concepts discovered
concepts = get_concepts_from_svd_kmeans(tfidf, svd, kmeans)


# Choose number of concepts to view
num_concepts = 10
# Choose number of terms to view for each concept
num_terms = 12
# Choose number of tweets to sample from each cluster centered on a concept
num_samples = 5


# Print top terms of the LSA concepts
print("\nTerms in LSA concepts:")
print_top_terms(get_concepts_from_svd(tfidf, svd), num_terms, num_concepts)

# Print top terms of the cluster centers from the largest clusters
print("\nTerms in cluster centers:")
print_top_terms(concepts, num_terms, num_concepts)

# Print a sample of tweets from the largest clusters
df = pd.DataFrame(raw_twitter_to_english_text(path_to_raw_data), columns=['text'])
df['label'] = pipeline.predict(sparse)
print_samples_of_text_by_label(df, num_concepts, num_samples)
