import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import numpy as np
import pandas as pd
import umap
import warnings
from sklearn.decomposition import PCA

# Constants (adjusted for visibility and scaling)
N_NEIGHBORS = 8
MIN_DIST = 0.1
RANDOM_STATE = 42
PLOT_SIZE = (10, 8)  # No change, size is reasonable
ALPHA = 0.8  # Slightly increase transparency
MARKER_SIZE = 10  # Increase marker size for better visibility
ZOOM_FACTOR = 1.2  # Set to 1 for no zoom or remove ZOOM_FACTOR usage

warnings.filterwarnings("ignore", category=UserWarning, module='umap')
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmbeddingVisualizer:
    def __init__(self, dataset_path, result_image_path, label_names={0: "Llama3", 1: "Gemma2", 2: "Phi3"}):
        self.dataset_path = dataset_path
        self.result_image_path = result_image_path
        
        self.df = None
        self.embeddings = None
        self.labels = None
        self.embedding_umap = None
        self.label_names = label_names

    def load_data(self):
        pd.set_option('display.max_columns', None)
        self.df = pd.read_csv(self.dataset_path)
        logging.info(f"Data loaded with shape {self.df.shape}")
        logging.info(f"Columns: {self.df.columns}")

    def get_embedding(self, input_str):
        input_str = input_str[2:-2]
        return [float(item) for item in input_str.split(", ")]
    
    def apply_umap(self, n_components):
        umap_model = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=n_components, random_state=RANDOM_STATE)
        self.embedding_umap = umap_model.fit_transform(self.embeddings)
        logging.info(f"UMAP applied in {n_components}D, resulting in shape {self.embedding_umap.shape}")

    ########
    # Process embeddings with padding 
    ########
    def expand_embedding_column(self, df, col_name):
        return pd.DataFrame(df[col_name].tolist(), index=df.index)

    def pad_embeddings(self, df, max_len):
        return pd.DataFrame(np.array([np.pad(embedding, (0, max_len - len(embedding)), 'constant') for embedding in df.values]))

    def process_embeddings_padding(self):
        # Extract and process embeddings
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

        # Set the labels
        df_llama_padded['label'] = 0
        df_gemma_padded['label'] = 1
        df_phi3_padded['label'] = 2

        df_combined = pd.concat([df_llama_padded, df_gemma_padded, df_phi3_padded], ignore_index=True)
        logging.info(f"Combined dataset created with shape {df_combined.shape}")

        self.embeddings = df_combined.drop(columns=['label']).values
        self.labels = df_combined['label'].values
        
        # Initialize label names here after processing embeddings
        #self.label_names = {0: "Llama3", 1: "Gemma2", 2: "Phi3"}
        logging.info(f"Label names set to {self.label_names}")
        logging.info(f"Padding applied. Reduced embeddings shape: {self.embeddings.shape}")

    #######    
    # Transform embeddings with PCA      
    #######
    def process_embeddings_PCA(self, target_dim):
        """
        Reduce dimensionality using PCA to a target dimension and return the embeddings and labels.
        """
        # Extract and process embeddings
        self.df['ta_embd_llama3'] = self.df['true_answer_embedding_llama3.1'].apply(self.get_embedding)
        self.df['ta_embd_gemma2'] = self.df['true_answer_embedding_gemma2'].apply(self.get_embedding)
        self.df['ta_embd_phi3'] = self.df['true_answer_embedding_phi3'].apply(self.get_embedding)

        # Drop the original embedding columns
        self.df = self.df.drop(columns=['true_answer_embedding_gemma2', 'true_answer_embedding_llama3.1', 'true_answer_embedding_phi3'])

        # Initialize lists to hold results
        reduced_embeddings = []
        labels = []
        
        # Process each embedding type
        for i, (embedding_col, label) in enumerate(zip(['ta_embd_llama3', 'ta_embd_gemma2', 'ta_embd_phi3'], [0, 1, 2])):
            # Expand the current embedding column
            df_expanded = self.expand_embedding_column(self.df[[embedding_col]], embedding_col)

            # Extract embeddings
            embeddings = df_expanded.values
            pca = PCA(n_components=target_dim)
            reduced = pca.fit_transform(embeddings)
                
            # Append reduced embeddings and corresponding labels
            reduced_embeddings.append(reduced)
            labels.extend([label] * reduced.shape[0])  # Extend labels for the number of embeddings

        # Concatenate all reduced embeddings into a single array        
        self.embeddings = np.vstack(reduced_embeddings)
        self.labels = np.array(labels)
        
        # Initialize label names here after processing embeddings
        #self.label_names = {0: "Llama3", 1: "Gemma2", 2: "Phi3"}
        logging.info(f"Label names set to {self.label_names}")
        logging.info(f"PCA applied. Reduced embeddings shape: {self.embeddings.shape}")
        
    #######
    # Plot UMAP 2D
    #######
    def plot_umap_2d(self):
        plt.figure(figsize=PLOT_SIZE)

        # Create a colormap with distinct colors
        cmap = plt.cm.get_cmap('viridis', len(self.label_names))  # Consistent colormap for 2D and 3D

        # Scatter for each label dynamically using label names
        for label_value, label_name in self.label_names.items():
            plt.scatter(self.embedding_umap[self.labels == label_value, 0], 
                        self.embedding_umap[self.labels == label_value, 1],
                        label=label_name, alpha=ALPHA, s=MARKER_SIZE, c=cmap(label_value))

        # Automatically adjust axis limits for better visualization
        plt.xlim(np.min(self.embedding_umap[:, 0]) * ZOOM_FACTOR, np.max(self.embedding_umap[:, 0]) * ZOOM_FACTOR)
        plt.ylim(np.min(self.embedding_umap[:, 1]) * ZOOM_FACTOR, np.max(self.embedding_umap[:, 1]) * ZOOM_FACTOR)

        # Add title, labels, and legend
        plt.title("2D UMAP Projection of Embeddings", fontsize=16)
        plt.xlabel("UMAP 1", fontsize=12)
        plt.ylabel("UMAP 2", fontsize=12)
        plt.legend()

        # Save and show the plot
        plt.savefig(self.result_image_path)
        logging.info(f"2D UMAP plot saved at {self.result_image_path}")
        plt.show()

    #######
    # Plot UMAP 3D
    #######
    def plot_umap_3d(self):
        fig = plt.figure(figsize=PLOT_SIZE)
        ax = fig.add_subplot(111, projection='3d')

        # Create the same colormap for consistency
        cmap = plt.cm.get_cmap('viridis', len(self.label_names))

        # Scatter for each label dynamically using label names
        for label_value, label_name in self.label_names.items():
            ax.scatter(self.embedding_umap[self.labels == label_value, 0], 
                       self.embedding_umap[self.labels == label_value, 1], 
                       self.embedding_umap[self.labels == label_value, 2],
                       label=label_name, alpha=ALPHA, s=MARKER_SIZE * 1.5, c=cmap(label_value))

        # Adjust axis scaling automatically to fit the data
        ax.auto_scale_xyz([np.min(self.embedding_umap[:, 0]) * ZOOM_FACTOR, np.max(self.embedding_umap[:, 0]) * ZOOM_FACTOR],
                          [np.min(self.embedding_umap[:, 1]) * ZOOM_FACTOR, np.max(self.embedding_umap[:, 1]) * ZOOM_FACTOR],
                          [np.min(self.embedding_umap[:, 2]) * ZOOM_FACTOR, np.max(self.embedding_umap[:, 2]) * ZOOM_FACTOR])

        # Add title, axis labels, and legend
        ax.set_title("3D UMAP Projection of Embeddings", fontsize=16)
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_zlabel("UMAP 3", fontsize=12)
        ax.legend()

        # Save the plot and display
        plt.savefig(self.result_image_path)
        logging.info(f"3D UMAP plot saved at {self.result_image_path}")
        plt.show()
        
    #######
    # Plot UMAP 3D
    #######    
    def plot_umap_8d_in_2d(self):
        num_dimensions = 8
        if self.embedding_umap.shape[1] < num_dimensions:
            logging.error(f"UMAP embeddings do not have {num_dimensions} dimensions!")
            return

        # Create pairwise plots (dimension 1 vs 2, 3 vs 4, etc.)
        num_pairs = num_dimensions // 2
        fig, axes = plt.subplots(nrows=num_pairs, ncols=1, figsize=(10, 4 * num_pairs))

        # Create a colormap
        cmap = plt.cm.get_cmap('viridis', len(self.label_names))

        for i in range(num_pairs):
            ax = axes[i]
            dim1 = 2 * i
            dim2 = 2 * i + 1

            # Scatter for each label dynamically using label names
            for label_value, label_name in self.label_names.items():
                ax.scatter(self.embedding_umap[self.labels == label_value, dim1], 
                           self.embedding_umap[self.labels == label_value, dim2],
                           label=label_name, alpha=ALPHA, s=MARKER_SIZE, c=cmap(label_value))

            ax.set_title(f"UMAP Projection (Dim {dim1+1} vs Dim {dim2+1})", fontsize=14)
            ax.set_xlabel(f"UMAP {dim1+1}")
            ax.set_ylabel(f"UMAP {dim2+1}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.result_image_path)
        logging.info(f"8D UMAP pairwise plot saved at {self.result_image_path}")
        plt.show()    
    
        
    #######
    # Plot UMAP 16D in 3D projections
    #######
    def plot_umap_16d_in_3d(self):
        num_dimensions = 16
        if self.embedding_umap.shape[1] < num_dimensions:
            logging.error(f"UMAP embeddings do not have {num_dimensions} dimensions!")
            return

        # We will group dimensions into sets of 3 to create multiple 3D plots
        num_triplets = num_dimensions // 3
        fig = plt.figure(figsize=(15, 5 * num_triplets))

        # Create a colormap
        cmap = plt.cm.get_cmap('viridis', len(self.label_names))

        # Generate subplots for each triplet of dimensions
        for i in range(num_triplets):
            ax = fig.add_subplot(num_triplets, 1, i+1, projection='3d')
            dim1 = 3 * i
            dim2 = 3 * i + 1
            dim3 = 3 * i + 2

            # Plot points for each label
            for label_value, label_name in self.label_names.items():
                ax.scatter(self.embedding_umap[self.labels == label_value, dim1],
                           self.embedding_umap[self.labels == label_value, dim2],
                           self.embedding_umap[self.labels == label_value, dim3],
                           label=label_name, alpha=ALPHA, s=MARKER_SIZE, c=cmap(label_value))

            ax.set_title(f"UMAP Projection (Dims {dim1+1}, {dim2+1}, {dim3+1})", fontsize=14)
            ax.set_xlabel(f"UMAP {dim1+1}")
            ax.set_ylabel(f"UMAP {dim2+1}")
            ax.set_zlabel(f"UMAP {dim3+1}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.result_image_path)
        logging.info(f"16D UMAP 3D projection plot saved at {self.result_image_path}")
        plt.show()    


if __name__ == "__main__":
    vis = EmbeddingVisualizer('./data/Natural-Questions-LLM-responses-400.csv', './data/results')
    vis.load_data()
    vis.process_embeddings_PCA(target_dim=50)
    vis.apply_umap(n_components=2)
    vis.plot_umap_2d()