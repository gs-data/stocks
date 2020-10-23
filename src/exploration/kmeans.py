if __name__ == '__main__':
    import os
    import sys
    sys.path[0] = os.getcwd()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

from src.data.raw_twitter_to_en_text import raw_twitter_to_english_text
from src.util.nlp_concepts import get_concepts_from_kmeans
from src.util.nlp_concepts import print_top_terms
from src.util.nlp_concepts import print_samples_of_text_by_label


path_to_raw_data = './data/raw/twitter_stream.txt'

# Create document term tfidf sparse matrix
tfidf = TfidfVectorizer(ngram_range=(1,2),
                        strip_accents='unicode',
                        stop_words='english')
sparse = tfidf.fit_transform(raw_twitter_to_english_text(path_to_raw_data))
print("Document-Term matrix size:", sparse.shape)


# Choose number of clusters
num_clusters = 50


# Compute clusters
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(sparse)

# Measure effectiveness of clusters
print("Iterations to convergence:", kmeans.n_iter_)
if (kmeans.get_params()['max_iter'] <= kmeans.n_iter_):
    print("CONVERGENCE WARNING: Iterations maxed out!")
print("Inertia:", kmeans.inertia_)

# Get cluster centers as possible concepts discovered
concepts = get_concepts_from_kmeans(tfidf, kmeans)


# Choose number of concepts to view
num_concepts = 25
# Choose number of terms to view for each concept
num_terms = 12
# Choose number of tweets to sample from each cluster centered on a concept
num_samples = 5


# Print top terms of the concepts from the largest clusters
print_top_terms(concepts, num_terms, num_concepts)

# Print a sample of tweets from the largest clusters
df = pd.DataFrame(raw_twitter_to_english_text(path_to_raw_data), columns=['text'])
df['label'] = kmeans.predict(sparse)
print_samples_of_text_by_label(df, num_concepts, num_samples)
