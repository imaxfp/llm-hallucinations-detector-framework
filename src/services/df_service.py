import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List
import logging
import pandas as pd 
from src.services.embedding_visualizer_service import EmbeddingVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_all_dfs_by_uid(dfs: list) -> pd.DataFrame:        
    # Find common uids across all DataFrames
    common_uids = set(dfs[0]['uid']).intersection(*(df['uid'] for df in dfs[1:]))
    # Filter each DataFrame to retain only rows with common uids and concatenate them
    filtered_dfs = [df[df['uid'].isin(common_uids)] for df in dfs]
    
    return filtered_dfs