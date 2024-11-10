import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List
import logging
import pandas as pd 
from src.services.embedding_visualizer_service import EmbeddingVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def centroids_df_to_dict(df: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Args:
        df (pd.DataFrame): DataFrame with columns representing UMAP coordinates for different sets.

    Returns:
        dict: Dictionary with each column name as a key and a tuple (x, y) as the coordinates.
    """
    centroids = {}
    for index, row in df.iterrows():
        centroids[index] = row.tolist()
    return centroids

# Euclidean distance function
def euclidean_distance(point1: List[float], point2: List[float]) -> float:    
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def euclidean_distance_dots(dots: Dict[str, List[float]]) -> Dict[str, list[float]]:
    """
    Calculates the Euclidean distance between each pair of points.

    (a,b,c)
    a->b
    a->c
    b->c
    
    Args:
        dots: A dictionary of point names as keys and (x, y) tuples as values.
    
    Returns:
        A dictionary with keys as point pairs (e.g., "a-b") and values as distances.
    """
    compar_dict = {}
    dot_keys = list(dots.keys())
    
    for i in range(len(dot_keys)):
        for j in range(i + 1, len(dot_keys)):
            point1, point2 = dot_keys[i], dot_keys[j]
            pair_key = f"{point1}--->{point2}"
            compar_dict[pair_key] = euclidean_distance(dots[point1], dots[point2])
    
    return compar_dict


def mesure_similarity(vis: EmbeddingVisualizer, experiment = 2, print_data = 1, shape = ''):
    logging.info(f"Data shape for similarity measurement = {shape}")        
    euclidean_distances = {}
    for i in range(experiment+1):
        df_umap = vis.apply_umap(n_components=2, rand_state=random.randint(0, 10**9))
        centroids_df = vis.find_centroids(df_umap)
        centroids_dict = centroids_df_to_dict(centroids_df)
        distance = euclidean_distance_dots(centroids_dict)

        for k, v in distance.items(): 
            tmp = euclidean_distances.get(k, [])
            tmp.append(v)
            euclidean_distances[k] = tmp
        if i != 0 and i % print_data == 0:
            #print results 
            average_distances = {k: round(sum(v) / len(v), 4) for k, v in euclidean_distances.items()}
            vis.plot_average_similarities_result(f'Centroids average distances for different randomers initialisation. Steps = {i} .Test data shape = {shape}', average_distances)
