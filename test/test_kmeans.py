# Write your k-means unit tests here
import pytest
import numpy as np
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from sklearn.cluster import KMeans as realkm
from cluster.utils import make_clusters

def test_errors():
    test_X = np.array([[1,2,3],[4,5,6],[7,8,9]])
    test_y = np.array([[1,2],[2,3]])

    with pytest.raises(ValueError) as err:
        KMeans(0)
    assert str(err.value) == "invalid input for k"
    with pytest.raises(ValueError) as err:
        KMeans(1.5)
    assert str(err.value) == "invalid input for k"
    with pytest.raises(ValueError) as err:
        KMeans(5, max_iter = -1)
    assert str(err.value) == "invalid input for max_iter"
    with pytest.raises(ValueError) as err:
        KMeans(5, tol = -2)
    assert str(err.value) == "invalid input for tol"
    
    tester = KMeans(2)
    tester.fit(test_X)
    with pytest.raises(ValueError) as err:
        tester.predict(test_y)
    assert str(err.value) == "points do not have same dimensions as training set"
    

def test_initialization():
    test_X = np.array([[1,2,3],[4,5,6]])
    test_kmean = KMeans(2)
    test_kmean.fit(test_X)
    assert len(test_kmean.get_centroids()) == 2

def test_kmeans():
    clusters, labels = make_clusters(k=4, scale=0.3)
    my_km = KMeans(k=4)
    my_km.fit(clusters)
    pred = my_km.predict(clusters)

    real_km = realkm(n_clusters=4)
    real_pred = real_km.fit_predict(clusters)

    translate_my = _standardize_output_4clusts(pred)
    translate_real = _standardize_output_4clusts(real_pred)

    assert translate_my == translate_real

def _standardize_output_4clusts(cluster_ident):
    # the nummber cluster label might not be the same between my version and the sklearn version.
    # this is an attempt to translate the cluster names to be comparable
    unique_vals = list(dict.fromkeys(cluster_ident))
    translate_dict = {unique_vals[0]:"a", unique_vals[1]:"b", unique_vals[2]:"c", unique_vals[3]:"d"}
    return [translate_dict.get(clust) for clust in cluster_ident]


