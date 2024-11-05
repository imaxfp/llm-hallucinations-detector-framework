import math
import numpy as np
import pandas as pd
from typing import Dict, List


def find_centroids(df):
    """
    Calculate the centroid for each column of coordinates in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns representing UMAP coordinates for different sets.

    Returns:
        pd.DataFrame: DataFrame with only x, y coordinates for each centroid.
    """
    centroids = {}

    for column in df.columns:
        # Stack coordinates for each column and calculate the mean
        coords = np.vstack(df[column].values)
        centroids[column] = coords.mean(axis=0)  # Get the centroid as [x, y]

    # Return a DataFrame with each row containing x, y for each centroid
    centroids_df = pd.DataFrame(centroids).T  # Transpose to have columns as rows
    centroids_df.columns = ['x', 'y']  # Name the columns for clarity
    return centroids_df


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