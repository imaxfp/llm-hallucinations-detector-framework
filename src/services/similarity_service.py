import numpy as np
import pandas as pd

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