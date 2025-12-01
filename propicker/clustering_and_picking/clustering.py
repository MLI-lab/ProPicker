"""
These functions are copied from DeePict
https://github.com/ZauggGroup/DeePiCt/blob/main/3d_cnn/src/tomogram_utils/coordinates_toolbox/clustering.py
"""

import numpy as np
from skimage import morphology as morph
from skimage.measure import regionprops_table
import pandas as pd


def get_clusters_within_size_range(binary_locmap: np.array, min_cluster_size: int,
                                   max_cluster_size, connectivity=1):
    if max_cluster_size is None:
        max_cluster_size = np.inf
    assert min_cluster_size <= max_cluster_size

    # find clusters and label them. Each cluster is assigned a unique integer from 0 to num_clusters-1
    # for example: [... 0   0   0   0   0   0   0   0   0 592 592 592 592 592   0   0   0   0 ...]
    labeled_clusters, num = morph.label(binary_locmap,
                                        background=0,
                                        return_num=True,
                                        connectivity=connectivity)
    labels_list, cluster_size = np.unique(labeled_clusters, return_counts=True)
    # excluding the background cluster: (e.g. where labels_list is zero)
    labels_list, cluster_size = labels_list[1:], cluster_size[1:]
    #print("cluster_sizes:", cluster_size)
    #print("number of clusters before size filtering = ", len(labels_list))
    #print("size range before size filtering: ", np.min(cluster_size), "to", maximum)
    labels_list_within_range = labels_list[(cluster_size > min_cluster_size) & (
            cluster_size <= max_cluster_size)]
    cluster_size_within_range = list(
        cluster_size[(cluster_size > min_cluster_size) & (
                cluster_size <= max_cluster_size)])
    return labeled_clusters, labels_list_within_range, cluster_size_within_range


def get_cluster_centroids(binary_locmap: np.array, min_cluster_size=1,
                          max_cluster_size=np.inf, connectivity=1) -> tuple:
    labeled_clusters, labels_list_within_range, cluster_size_within_range = \
        get_clusters_within_size_range(binary_locmap=binary_locmap,
                                       min_cluster_size=min_cluster_size,
                                       max_cluster_size=max_cluster_size,
                                       connectivity=connectivity)
    # Create binary mask of the labels within range
    clusters_map_in_range = np.zeros(labeled_clusters.shape)
    clusters_map_in_range[np.isin(labeled_clusters, labels_list_within_range)] = 1
    # Find out the centroids of the labels within range
    filtered_labeled_clusters = (labeled_clusters * clusters_map_in_range).astype(np.int32)
    props = regionprops_table(filtered_labeled_clusters, properties=('label', 'centroid'))
    centroids_list = [np.rint([x, y, z]) for _, x, y, z in sorted(zip(props['label'].tolist(),
                                                                      props['centroid-0'].tolist(),
                                                                      props['centroid-1'].tolist(),
                                                                      props['centroid-2'].tolist()))]
    return clusters_map_in_range, centroids_list, cluster_size_within_range

def get_cluster_centroids_df(binary_locmap, min_cluster_size=1, max_cluster_size=np.inf, connectivity=1):
    clusters_labeled_by_size, centroids_list, cluster_size_list = get_cluster_centroids(
            binary_locmap=binary_locmap,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            connectivity=1
    )

    df = pd.DataFrame(columns=["X", "Y", "Z", "size"])
    df["X"] = [c[2] for c in centroids_list]
    df["Y"] = [c[1] for c in centroids_list]
    df["Z"] = [c[0] for c in centroids_list]
    df["size"] = cluster_size_list
    return df