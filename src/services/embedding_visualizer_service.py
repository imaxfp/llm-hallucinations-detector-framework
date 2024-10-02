import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import numpy as np
import pandas as pd
import umap
import warnings

# Global Constants
N_NEIGHBORS = 8
MIN_DIST = 0.1
RANDOM_STATE = 42
PLOT_SIZE = (10, 8)
ALPHA = 0.7
MARKER_SIZE = 20  # Reduce marker size
ZOOM_FACTOR = 0.9  # Adjust zoom level (closer to 1 is less zoom, <1 is more zoom)

warnings.filterwarnings("ignore", category=UserWarning, module='umap')
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmbeddingVisualizer:
    def __init__(self, dataset_path, result_image_path):
        self.dataset_path = dataset_path
        self.result_image_path = result_image_path
        
        self.df = None
        self.embeddings = None
        self.labels = None
        self.embedding_2d = None
        self.embedding_3d = None

    def load_data(self):
        pd.set_option('display.max_columns', None)
        self.df = pd.read_csv(self.dataset_path)
        logging.info(f"Data loaded with shape {self.df.shape}")
        logging.info(f"Columns: {self.df.columns}")

    def get_embedding(self, input_str):
        input_str = input_str[2:-2]
        return [float(item) for item in input_str.split(", ")]

    def process_embeddings(self):
        self.df['ta_embd_llama3'] = self.df['true_answer_embedding_llama3.1'].apply(self.get_embedding)
        self.df['ta_embd_gemma2'] = self.df['true_answer_embedding_gemma2'].apply(self.get_embedding)
        self.df['ta_embd_phi3'] = self.df['true_answer_embedding_phi3'].apply(self.get_embedding)
        self.df = self.df.drop(columns=['true_answer_embedding_gemma2', 'true_answer_embedding_llama3.1', 'true_answer_embedding_phi3'])
        
        df_llama = self.df[['ta_embd_llama3']]
        df_gemma = self.df[['ta_embd_gemma2']]
        df_phi3 = self.df[['ta_embd_phi3']]

        df_llama_expanded = self.expand_embedding_column(df_llama, 'ta_embd_llama3')
        df_gemma_expanded = self.expand_embedding_column(df_gemma, 'ta_embd_gemma2')
        df_phi3_expanded = self.expand_embedding_column(df_phi3, 'ta_embd_phi3')

        max_len = max(df_llama_expanded.shape[1], df_gemma_expanded.shape[1], df_phi3_expanded.shape[1])
        
        df_llama_padded = self.pad_embeddings(df_llama_expanded, max_len)
        df_gemma_padded = self.pad_embeddings(df_gemma_expanded, max_len)
        df_phi3_padded = self.pad_embeddings(df_phi3_expanded, max_len)

        df_llama_padded['label'] = 0
        df_gemma_padded['label'] = 1
        df_phi3_padded['label'] = 2

        df_combined = pd.concat([df_llama_padded, df_gemma_padded, df_phi3_padded], ignore_index=True)
        logging.info(f"Combined dataset created with shape {df_combined.shape}")

        self.embeddings = df_combined.drop(columns=['label']).values
        self.labels = df_combined['label'].values

    def expand_embedding_column(self, df, col_name):
        return pd.DataFrame(df[col_name].tolist(), index=df.index)

    def pad_embeddings(self, df, max_len):
        return pd.DataFrame(np.array([np.pad(embedding, (0, max_len - len(embedding)), 'constant') for embedding in df.values]))

    def apply_umap_2d(self):
        umap_model = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=2, random_state=RANDOM_STATE)
        self.embedding_2d = umap_model.fit_transform(self.embeddings)
        logging.info(f"UMAP applied in 2D, resulting in shape {self.embedding_2d.shape}")

    def apply_umap_3d(self):
        umap_model = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=3, random_state=RANDOM_STATE)
        self.embedding_3d = umap_model.fit_transform(self.embeddings)
        logging.info(f"UMAP applied in 3D, resulting in shape {self.embedding_3d.shape}")

    def plot_umap_2d(self):
        plt.figure(figsize=PLOT_SIZE)
        plt.scatter(self.embedding_2d[self.labels == 0, 0], self.embedding_2d[self.labels == 0, 1], label="Llama3", alpha=ALPHA, s=MARKER_SIZE, c='r')
        plt.scatter(self.embedding_2d[self.labels == 1, 0], self.embedding_2d[self.labels == 1, 1], label="Gemma2", alpha=ALPHA, s=MARKER_SIZE, c='g')
        plt.scatter(self.embedding_2d[self.labels == 2, 0], self.embedding_2d[self.labels == 2, 1], label="Phi3", alpha=ALPHA, s=MARKER_SIZE, c='b')

        plt.xlim(np.min(self.embedding_2d[:, 0]) * ZOOM_FACTOR, np.max(self.embedding_2d[:, 0]) * ZOOM_FACTOR)
        plt.ylim(np.min(self.embedding_2d[:, 1]) * ZOOM_FACTOR, np.max(self.embedding_2d[:, 1]) * ZOOM_FACTOR)

        plt.title("2D UMAP Projection of Embeddings from Llama, Gemma, and Phi Models", fontsize=16)
        plt.xlabel("UMAP 1", fontsize=12)
        plt.ylabel("UMAP 2", fontsize=12)
        plt.legend()
        
        plt.savefig(self.result_image_path)
        logging.info(f"2D UMAP plot saved at {self.result_image_path}")
        plt.show()

    def plot_umap_3d(self):
        fig = plt.figure(figsize=PLOT_SIZE)
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.embedding_3d[self.labels == 0, 0], self.embedding_3d[self.labels == 0, 1], self.embedding_3d[self.labels == 0, 2],
                   label="Llama3", alpha=ALPHA, s=MARKER_SIZE, c='r', marker='o')
        ax.scatter(self.embedding_3d[self.labels == 1, 0], self.embedding_3d[self.labels == 1, 1], self.embedding_3d[self.labels == 1, 2],
                   label="Gemma2", alpha=ALPHA, s=MARKER_SIZE, c='g', marker='^')
        ax.scatter(self.embedding_3d[self.labels == 2, 0], self.embedding_3d[self.labels == 2, 1], self.embedding_3d[self.labels == 2, 2],
                   label="Phi3", alpha=ALPHA, s=MARKER_SIZE, c='b', marker='s')

        ax.set_xlim(np.min(self.embedding_3d[:, 0]) * ZOOM_FACTOR, np.max(self.embedding_3d[:, 0]) * ZOOM_FACTOR)
        ax.set_ylim(np.min(self.embedding_3d[:, 1]) * ZOOM_FACTOR, np.max(self.embedding_3d[:, 1]) * ZOOM_FACTOR)
        ax.set_zlim(np.min(self.embedding_3d[:, 2]) * ZOOM_FACTOR, np.max(self.embedding_3d[:, 2]) * ZOOM_FACTOR)

        ax.set_title("3D UMAP Projection of Embeddings from Llama, Gemma, and Phi Models", fontsize=16)
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_zlabel("UMAP 3", fontsize=12)
        ax.legend()

        plt.savefig(self.result_image_path)
        logging.info(f"3D UMAP plot saved at {self.result_image_path}")
        plt.show()
        
    def apply_umap_16d(self):
        """
        Apply UMAP with 16 dimensions on the embeddings.
        """
        umap_model = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=16, random_state=RANDOM_STATE)
        self.embedding_16d = umap_model.fit_transform(self.embeddings)
        logging.info(f"UMAP applied in 16D, resulting in shape {self.embedding_16d.shape}")
    
        
    def plot_pairwise_3d_from_16d(self):
        """
        Plot pairwise 3D projections from 16D UMAP embeddings.
        This will display 4 pairwise 3D plots.
        """
        fig = plt.figure(figsize=(18, 18))
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224, projection='3d')

        # Plotting pairwise combinations of the first 4 sets of dimensions
        ax1.scatter(self.embedding_16d[self.labels == 0, 0], self.embedding_16d[self.labels == 0, 1], self.embedding_16d[self.labels == 0, 2], label="Llama3", alpha=ALPHA, s=MARKER_SIZE, c='r')
        ax1.scatter(self.embedding_16d[self.labels == 1, 0], self.embedding_16d[self.labels == 1, 1], self.embedding_16d[self.labels == 1, 2], label="Gemma2", alpha=ALPHA, s=MARKER_SIZE, c='g')
        ax1.scatter(self.embedding_16d[self.labels == 2, 0], self.embedding_16d[self.labels == 2, 1], self.embedding_16d[self.labels == 2, 2], label="Phi3", alpha=ALPHA, s=MARKER_SIZE, c='b')
        ax1.set_title("UMAP Dimension 1, 2, and 3")
        ax1.set_xlabel("UMAP Dimension 1")
        ax1.set_ylabel("UMAP Dimension 2")
        ax1.set_zlabel("UMAP Dimension 3")
        ax1.legend()

        ax2.scatter(self.embedding_16d[self.labels == 0, 3], self.embedding_16d[self.labels == 0, 4], self.embedding_16d[self.labels == 0, 5], label="Llama3", alpha=ALPHA, s=MARKER_SIZE, c='r')
        ax2.scatter(self.embedding_16d[self.labels == 1, 3], self.embedding_16d[self.labels == 1, 4], self.embedding_16d[self.labels == 1, 5], label="Gemma2", alpha=ALPHA, s=MARKER_SIZE, c='g')
        ax2.scatter(self.embedding_16d[self.labels == 2, 3], self.embedding_16d[self.labels == 2, 4], self.embedding_16d[self.labels == 2, 5], label="Phi3", alpha=ALPHA, s=MARKER_SIZE, c='b')
        ax2.set_title("UMAP Dimension 4, 5, and 6")
        ax2.set_xlabel("UMAP Dimension 4")
        ax2.set_ylabel("UMAP Dimension 5")
        ax2.set_zlabel("UMAP Dimension 6")
        ax2.legend()

        ax3.scatter(self.embedding_16d[self.labels == 0, 6], self.embedding_16d[self.labels == 0, 7], self.embedding_16d[self.labels == 0, 8], label="Llama3", alpha=ALPHA, s=MARKER_SIZE, c='r')
        ax3.scatter(self.embedding_16d[self.labels == 1, 6], self.embedding_16d[self.labels == 1, 7], self.embedding_16d[self.labels == 1, 8], label="Gemma2", alpha=ALPHA, s=MARKER_SIZE, c='g')
        ax3.scatter(self.embedding_16d[self.labels == 2, 6], self.embedding_16d[self.labels == 2, 7], self.embedding_16d[self.labels == 2, 8], label="Phi3", alpha=ALPHA, s=MARKER_SIZE, c='b')
        ax3.set_title("UMAP Dimension 7, 8, and 9")
        ax3.set_xlabel("UMAP Dimension 7")
        ax3.set_ylabel("UMAP Dimension 8")
        ax3.set_zlabel("UMAP Dimension 9")
        ax3.legend()

        ax4.scatter(self.embedding_16d[self.labels == 0, 9], self.embedding_16d[self.labels == 0, 10], self.embedding_16d[self.labels == 0, 11], label="Llama3", alpha=ALPHA, s=MARKER_SIZE, c='r')
        ax4.scatter(self.embedding_16d[self.labels == 1, 9], self.embedding_16d[self.labels == 1, 10], self.embedding_16d[self.labels == 1, 11], label="Gemma2", alpha=ALPHA, s=MARKER_SIZE, c='g')
        ax4.scatter(self.embedding_16d[self.labels == 2, 9], self.embedding_16d[self.labels == 2, 10], self.embedding_16d[self.labels == 2, 11], label="Phi3", alpha=ALPHA, s=MARKER_SIZE, c='b')
        ax4.set_title("UMAP Dimension 10, 11, and 12")
        ax4.set_xlabel("UMAP Dimension 10")
        ax4.set_ylabel("UMAP Dimension 11")
        ax4.set_zlabel("UMAP Dimension 12")
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.result_image_path)
        logging.info(f"Pairwise 3D plots from 16D UMAP saved at {self.result_image_path}")
        plt.show()

        
