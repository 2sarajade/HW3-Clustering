# write your silhouette score unit tests here
import pytest
import numpy as np
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from sklearn.metrics import silhouette_score
from cluster.utils import make_clusters

def test_errors():
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = [1,1,1,2,2,2]
    with pytest.raises(ValueError) as err:
        Silhouette().score(clusters, pred)
    assert str(err.value) == "Dimensionsn of inputs do not match"
    

def test_silhouette():
    clusters, labels = make_clusters(k=4, scale=0.3)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)

    real_scores = silhouette_score(clusters, pred, metric='euclidean')

    assert np.mean(scores) == real_scores