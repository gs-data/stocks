import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

tfidf = TfidfVectorizer(ngram_range=(1,2),
                        strip_accents='unicode')

path_to_processed_data = './data/processed/twitter_texts.txt'

with open(path_to_processed_data, 'r') as processed:
    sparse_matrix = tfidf.fit_transform(processed)

# svd = TruncatedSVD(n_components=100)
# clusters = [*range(10, 21)]
# inertias = []
# for n in clusters:
#     kmeans = KMeans(n_clusters=n)
#     pipeline = make_pipeline(svd, kmeans)
#     pipeline.fit(sparse_matrix)
#     inertias.append(kmeans.inertia_)

# elbow = pd.DataFrame({'clusters': clusters, 'inertia': inertias})
# elbow.plot(x='clusters', y='inertia')

svd = TruncatedSVD(n_components=100)
kmeans = KMeans(n_clusters=50)
pipeline = make_pipeline(svd, kmeans)

pipeline.fit(sparse_matrix)

labels = pipeline.predict(sparse_matrix)

with open(path_to_processed_data, 'r') as processed:
    tweets = processed.readlines()

df = pd.DataFrame({'label': labels, 'tweet': tweets})
df.tweet = df.tweet.str.rstrip('\n')

for row in df.groupby('label').head().sort_values('label').itertuples():
    print(row.label, row.tweet)

