import logging
import seaborn as sns
import matplotlib.pyplot as plt  # You still need matplotlib as a backend
import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
import umap as mp
import ast

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
#from tqdm import TqdmWarning

# Suppress TqdmWarnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore')
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingVisualizer:
    def __init__(self, df=None, dataset_path=None):
        self.dataset_path = dataset_path        
        self.df = df
        self.df_prepared = None
        self.df_umap_processed = None
        self.labels = None                 

    def load_data(self):
        pd.set_option('display.max_columns', None)
        self.df = pd.read_csv(self.dataset_path)
        logging.info(f"Data loaded with shape {self.df.shape}")
        logging.info(f"Columns: {self.df.columns}")        
        return self.df

    
    def convert_columns_to_float_arrays(self):                
        """Convert all columns of the DataFrame from string representations of lists to flat NumPy float arrays."""
        for col in self.df.columns:
            self.df[col] = self.df[col].apply(
                lambda sequence: np.array(ast.literal_eval(sequence), dtype=float).flatten() if isinstance(sequence, str) else sequence
            )
            

    def pad_to_max_len(self, sequence, max_len):
        # Pad or truncate the sequence to the desired length, and make sure it's a float array
        return np.pad(sequence[:max_len], (0, max(0, max_len - len(sequence))), mode='constant').astype(float)
 
    def process_embeddings_padding(self):
        """Process the DataFrame by converting string columns to float arrays and applying padding."""
        
        # Calculate the maximum length across all columns
        max_len = self.df.applymap(len).max().max()

        # Apply padding to all columns
        for col in self.df.columns:
            self.df[col] = self.df[col].apply(lambda x: self.pad_to_max_len(x, max_len))

        # Store the padded DataFrame
        self.df_prepared = self.df

        # Logging
        logging.info(f'Max length across specified columns: {max_len}')
        logging.info(self.df.columns.tolist())   
        logging.info(f"Padding applied. Reduced embeddings shape: {self.df_prepared.shape}")        
    
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


        # Create a new DataFrame to hold the UMAP results
        new_df = pd.DataFrame()

        # Create UMAP model
        umap_model = mp.UMAP(
            n_neighbors=N_NEIGHBORS,
            min_dist=MIN_DIST,
            n_components=n_components,
            random_state=RANDOM_STATE)
        
        for column in self.df_prepared:

            logging.debug(f"COLUMN DATA: {self.df_prepared[column]}")
            try:                                                            
                # Stack the arrays into a 2D array of shape (n_samples, n_features)
                #data_to_compress = np.stack(self.df_prepared[column].values)
                #column_data = self.df_prepared[column].values.reshape(-1, 1)

                column_data = np.stack(self.df_prepared[column].values)

                # Apply UMAP
                umap_transformed = umap_model.fit_transform(column_data)
                
                # Add the compressed data to the new DataFrame as a single column
                new_df[column + '_umap'] = [umap_transformed[i] for i in range(umap_transformed.shape[0])]
                
            except Exception as e:
                logging.info(f"An error occurred: {e}")     

        # Store the processed DataFrame
        self.df_umap_processed = new_df           
        return self.df_umap_processed

 


        
    #######
    # Plot UMAP 2D
    #######
    def plot_umap_2d(self, title=str(), save_path=None):
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
    


    #######    
    # Transform embeddings with PCA  
    # TODO Techniques like PCA or t-SNE     
    #######
    def process_embeddings_PCA(self, target_dim):
        """
        Reduce dimensionality using PCA to a target dimension and return the embeddings and labels.
        """
        # Extract and process embeddings
        self.df['ta_embd_llama3'] = self.df['true_answer_embedding_llama3.1']
        self.df['ta_embd_gemma2'] = self.df['true_answer_embedding_gemma2']
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


if __name__ == "__main__":    
    true_answer_columns = ['true_answer_embedding_llama3.1']    
    llm_generation_columns = [ 'embedding_llama3.1']      

    df_llm_res = pd.read_csv('./data/NQ-LLM-responses.csv')
    df_fabrication_hallucinations = pd.read_csv('./data/Fabrication-NQ-LLM-responses.csv')
    print(df_fabrication_hallucinations.columns)

    # Create a new DataFrame by selecting specific columns and renaming them
    new_df_llm_res = df_llm_res[['true_answer_embedding_llama3.1', 'embedding_llama3.1']].rename(
        columns={
            'true_answer_embedding_llama3.1': 'True_answer',
            'embedding_llama3.1': 'answer_llama3.1'
        }
    )
    # Create a new DataFrame by selecting specific columns and renaming them
    new_df_fabrication_hallucinations = df_llm_res[['embedding_llama3.1']].rename(
        columns={
            'embedding_llama3.1': 'answer_llama3.1'
        }
    )

    combined_df = pd.concat([new_df_llm_res, new_df_fabrication_hallucinations], axis=1)
    print(combined_df.columns)