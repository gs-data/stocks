import pandas as pd

def get_concepts_from_svd(tfidf, svd):
    """Get LSA/SVD "concepts" expressed as vectors in term space.
    
    Parameters
    ----------
    tfidf : TfidfVectorizer
        Fitted vectorizer with learned term vocabulary.
    svd : TruncatedSVD
        SVD fitted to document-term matrix returned by tfidf.
    
    Returns
    -------
    pandas.DataFrame
        Columns are terms, rows are "concepts" sorted descending 
        by magnitude of singular values.
    """
    return pd.DataFrame(svd.components_, columns=tfidf.get_feature_names())


def get_concepts_from_kmeans(tfidf, kmeans):
    """Get kmeans cluster centers in term space.
    
    Parameters
    ----------
    tfidf : TfidfVectorizer
        Fitted vectorizer with learned term vocabulary.
    kmeans : KMeans
        KMeans fitted to document-term matrix returned by tfidf.
    
    Returns
    -------
    pandas.DataFrame
        Columns are terms, rows are "concepts" sorted by cluster size.
    """
    df = pd.DataFrame(kmeans.cluster_centers_, columns=tfidf.get_feature_names())
    return df.reindex(pd.Series(kmeans.labels_).value_counts().index)


def get_concepts_from_svd_kmeans(tfidf, svd, kmeans):
    """Get kmeans cluster centers in SVD concept space expressed as vectors in term space.
    
    Parameters
    ----------
    tfidf : TfidfVectorizer
        Fitted vectorizer with learned term vocabulary.
    svd : TruncatedSVD
        SVD fitted to document-term matrix returned by tfidf.
    kmeans : KMeans
        KMeans fitted to document-concept matrix returned by svd.
    
    Returns
    -------
    pandas.DataFrame
        Columns are terms, rows are "concepts".
    """
    low_dim = pd.DataFrame(kmeans.cluster_centers_)
    low_dim = low_dim.reindex(pd.Series(kmeans.labels_).value_counts().index)
    return pd.DataFrame(svd.inverse_transform(low_dim), columns=tfidf.get_feature_names(), index=low_dim.index)


def get_concepts_from_nmf(tfidf, nmf):
    """Get NMF "concepts" expressed as vectors in term space.
    
    Parameters
    ----------
    tfidf : TfidfVectorizer
        Fitted vectorizer with learned term vocabulary.
    nmf : NMF
        NMF fitted to document-term matrix returned by tfidf.
    
    Returns
    -------
    pandas.DataFrame
        Columns are terms, rows are "concepts".
    """
    return pd.DataFrame(nmf.components_, columns=tfidf.get_feature_names())


def top_terms_of_concepts(concepts, num_terms, num_concepts):
    """Find top terms of vectors in term space
    
    Parameters
    ----------
    concepts : pandas.DataFrame
        Columns are terms, rows are "concepts" as vectors in term space.
        Rows should be sorted in descending order of importance.
    num_terms : int 
        Number of top terms to get for each concept.
    num_concepts : int
        Number of concepts to get terms from
    
    Returns
    -------
    list of list of str
        List for each concept a list of terms
    """
    return [[term for term in concepts.iloc[row].nlargest(num_terms).index] for row in range(num_concepts)]


def print_top_terms(concepts, num_terms, num_concepts):
    """Print top terms of concepts
    
    Parameters
    ----------
    concepts : pandas.DataFrame
        Columns are terms, rows are "concepts" as vectors in term space.
        Rows should be sorted in descending order of importance.
    num_terms : int 
        Number of top terms to get for each concept.
    num_concepts : int
        Number of concepts to get terms from
    """
    top_terms = top_terms_of_concepts(concepts, num_terms, num_concepts)
    print('\n'.join([str(concepts.index[i]) + ': ' + ', '.join(terms) for i, terms in enumerate(top_terms)]))


def print_samples_of_text_by_label(labeled_text, num_labels, num_samples):
    """Print random sample of documents from each label
    
    Parameters
    ----------
    labeled_text : pandas.DataFrame
        Rows are documents, should have 'text' column and 'label' column
    num_labels : int
        Number of labels from which to draw samples.
    num_samples : int
        Number of samples to draw from each label group.
    """
    label_counts = labeled_text.label.value_counts().nlargest(num_labels)
    groups = labeled_text.groupby('label')
    for label in label_counts.index:
        print(f'\nLabel {label} containing {label_counts[label]} samples:')
        print('\n'.join(groups.get_group(label).sample(num_samples).text))




