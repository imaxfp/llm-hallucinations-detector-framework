import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import numpy as np
import pandas as pd
import umap
import warnings
from sklearn.decomposition import PCA

# Constants (adjusted for visibility and scaling)
N_NEIGHBORS = 10  # Adjusted for better local/global balance
MIN_DIST = 0.2    # Slightly increased to avoid clustering all points together
RANDOM_STATE = 42  # Keep this for reproducibility
PLOT_SIZE = (10, 8)  # No change
ALPHA = 0.8  # Keep for transparency
MARKER_SIZE = 10  # No change, fine for visibility
ZOOM_FACTOR = 1.2  # Keep as is or adjust based on visualization needs

# Suppress UserWarnings
import warnings
from tqdm import TqdmWarning

# Suppress TqdmWarnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore')
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmbeddingVisualizer:
    def __init__(self, dataset_path, result_image_path):
        self.dataset_path = dataset_path
        self.result_image_path = result_image_path
        
        self.df = None
        self.df_prepared = None
        self.df_umap_processed = None
        self.labels = None
        

    def load_data(self):
        pd.set_option('display.max_columns', None)
        self.df = pd.read_csv(self.dataset_path)
        logging.info(f"Data loaded with shape {self.df.shape}")
        logging.info(f"Columns: {self.df.columns}")

    def get_embedding(self, input_str):
        input_str = input_str[2:-2]
        return [float(item) for item in input_str.split(", ")]
    
    def apply_umap(self, n_components):        
        """
        UMAP (Uniform Manifold Approximation and Projection) is designed to find meaningful low-dimensional representations
        of data by analyzing the relationships between multiple data points
        n_components : int
            The number of dimensions to reduce to.
        Returns:
        None
        """
        # Check if embeddings are not empty
        if self.df_prepared is None or self.df_prepared.size == 0:
            logging.error("Embeddings are empty. Cannot apply UMAP.")
            return None

        # Create UMAP model
        umap_model = umap.UMAP(
            n_neighbors=N_NEIGHBORS,
            min_dist=MIN_DIST,
            n_components=n_components,
            random_state=RANDOM_STATE)
        
        try:
            # Create a new DataFrame to hold the UMAP results
            new_df = pd.DataFrame(columns=self.df_prepared.columns)

            # Process each embedding entity separately
            for column in self.df_prepared.columns:
                # Stack the arrays of the current column into a 2D array
                data_to_compress = np.array(self.df_prepared[column].tolist())

                # Initialize UMAP
                umap_model = umap.UMAP(n_components=2)

                # Fit and transform the data for the current column
                compressed_data = umap_model.fit_transform(data_to_compress)

                # Add the compressed data to the new DataFrame
                new_df[column] = [compressed_data[i] for i in range(compressed_data.shape[0])]

            # Display the new DataFrame
            #logging.info(new_df)
            self.df_umap_processed = new_df

        except Exception as e:                       
            logging.info(f"An error occurred: {e}")       
        
        return self.df_umap_processed



    ########
    # Process embeddings with padding 
    ########
    def pad_embeddings(self, col, max_len):        
        """
        Pads each element of the input column to ensure they all have the same length.
        
        Args:
            col (pandas Series): The column to pad.
            max_len (int): The maximum length to pad to.

        Returns:
            pandas Series: A new Series where each list in the original column is padded with zeros to reach max_len.
        """        
        """Pads an individual embedding to max_len."""
        return col + [0] * (max_len - len(col)) if len(col) < max_len else col

 
    def process_embeddings_padding(self, true_answer_columns, llm_generation_columns):                                                          
        embedding_columns = true_answer_columns + llm_generation_columns        
        
        for col in embedding_columns:
            self.df[col] = self.df[col].apply(self.get_embedding)             
            
        # Delete all columns except those in the embedding_columns list
        cols_to_drop = self.df.columns.difference(embedding_columns)
        self.df.drop(columns=cols_to_drop, inplace=True)  
        
        # Calculate maximum lengths for each column in embedding_columns
        max_len = 0
        for col in self.df.columns:
            if col in embedding_columns:
                current_max_length = self.df[col].apply(len).max()
                max_len = max(max_len, current_max_length)
        #apply padding
        for col in self.df.columns:
            if col in embedding_columns:
                self.df[col] = self.df[col].apply(lambda x: self.pad_embeddings(x, max_len))
                

        # Print the maximum length found
        logging.info(f'Max length across specified columns: {max_len}')                  
        logging.info(self.df.columns.tolist())   
        
        # Apply pad_embeddings to all columns of the DataFrame
        self.df_prepared = self.df
                           
        # Initialize label names (if required)
        logging.info(f"Padding applied. Reduced embeddings shape: {self.df_prepared.shape}")



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
    def plot_umap_2d(self, title=str(), save_path='./data/results/umap_2d.png'):
        plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

        # Get the number of columns in the DataFrame
        num_columns = self.df_umap_processed.shape[1]  # Using the new DataFrame with UMAP processed data
        column_names = self.df_umap_processed.columns

        # Create a colormap with distinct colors based on the number of columns
        cmap = plt.cm.get_cmap('tab10', num_columns)  # 'tab10' provides up to 10 distinct colors

        # Define different markers for the objects
        markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'H', 'v', '<']  # Add more if needed
        num_markers = len(markers)

        # Plot each column's coordinates
        for i, column in enumerate(column_names):
            # Extract the UMAP coordinates for the current column
            umap_coords = np.array(self.df_umap_processed[column].tolist())            
            # Choose marker; wrap around if more columns than markers
            marker = markers[i % num_markers]            
            # Scatter plot for the current column with unique color and marker
            plt.scatter(umap_coords[:, 0], umap_coords[:, 1], label=column, alpha=0.7, s=20, color=cmap(i), marker=marker)

        # Add title, labels, and legend
        plt.title(title, fontsize=14)
        plt.xlabel("UMAP Component 1", fontsize=8)
        plt.ylabel("UMAP Component 2", fontsize=8)
        plt.legend(title="Objects", bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot

        # Automatically adjust the layout
        plt.tight_layout()
        # Save the plot to the specified path
        plt.savefig(save_path)
        # Show the plot
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
    


if __name__ == "__main__":
    
    true_answer_columns = ['true_answer_embedding_llama3.1', 'true_answer_embedding_gemma2', 'true_answer_embedding_phi3']    
    llm_generation_columns = [ 'embedding_llama3.1', 'embedding_gemma2', 'embedding_phi3']      
    
    vis = EmbeddingVisualizer('./data/Fabrication-NQ-LLM-responses.csv', './data/results')
    vis.load_data()
    #vis.process_embeddings_PCA(target_dim=10)
    vis.process_embeddings_padding(true_answer_columns, llm_generation_columns)
    vis.apply_umap(n_components=2)
    vis.plot_umap_2d()