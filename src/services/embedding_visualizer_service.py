import io
import logging
import seaborn as sns
import matplotlib.pyplot as plt  # You still need matplotlib as a backend
import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
import umap as mp
import ast
from PIL import Image

# Constants (adjusted for visibility and scaling)
N_NEIGHBORS = 10  # Adjusted for better local/global balance
MIN_DIST = 0.2    # Slightly increased to avoid clustering all points together
RANDOM_STATE = 17  # Keep this for reproducibility
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
    
    def apply_umap(self, n_components=2, rand_state=RANDOM_STATE):        
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
            random_state=rand_state)
        
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
        return new_df
    

    def process_embeddings_PCA(self, target_dim, df):
        """
        Reduce dimensionality of all columns in the provided DataFrame using PCA and return a new DataFrame.
        Parameters:
            target_dim (int): The target dimension for PCA.
            df (pd.DataFrame): The DataFrame containing the embeddings (each cell containing an array).    
        Returns:
            new_df (pd.DataFrame): A DataFrame containing the PCA-reduced embeddings with labeled columns.
        ------------------------------------
        Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms 
        a dataset to a lower-dimensional space while retaining as much variance as possible. 
        Here's a breakdown of the PCA process:

        1. **Standardize the Data**: Center the data by subtracting the mean of each feature. 
        This ensures that all features have a mean of zero, which is essential for PCA to work effectively.
        
        2. **Compute the Covariance Matrix**: Calculate the covariance matrix of the standardized data.
        The covariance matrix reveals the relationships between different features in terms of their variances 
        and how they co-vary with each other.
        
        3. **Calculate Eigenvalues and Eigenvectors**: Decompose the covariance matrix to get eigenvalues and 
        eigenvectors. Eigenvalues indicate the amount of variance captured by each principal component, 
        while eigenvectors indicate the direction of each component in the original feature space.
        
        4. **Sort and Select Principal Components**: Sort the eigenvalues in descending order and select the top 
        `target_dim` eigenvectors corresponding to the largest eigenvalues. These eigenvectors represent the 
        directions (principal components) that capture the most variance in the data.
        
        5. **Project the Data**: Transform the original data onto the selected principal components to reduce its 
        dimensionality. This results in a new representation of the data in a lower-dimensional space, where 
        each dimension is a principal component capturing the maximum possible variance.
        
        This method allows us to capture the most important patterns in the data while reducing noise and 
        computational complexity in higher-dimensional spaces.

        Main Logic:
        - Expand any list-like entries in the columns to separate dimensions.
        - Apply PCA on the expanded DataFrame to reduce the data to the desired number of dimensions.
        - Return a new DataFrame with the PCA-reduced embeddings.
        """
        # Expand each list in the columns to separate columns for each dimension in the embeddings
        expanded_df = pd.DataFrame()

        for col in df.columns:
            # Expand each list-like cell in the column into separate columns
            expanded_cols = pd.DataFrame(df[col].tolist(), index=df.index)
            # Rename columns with original column name as prefix
            expanded_cols.columns = [f"{col}_{i}" for i in range(expanded_cols.shape[1])]
            expanded_df = pd.concat([expanded_df, expanded_cols], axis=1)

        # Apply PCA to reduce dimensions
        pca = PCA(n_components=target_dim)
        reduced_embeddings = pca.fit_transform(expanded_df.values)
        
        # Create a new DataFrame to hold the PCA results with meaningful column names
        new_df = pd.DataFrame(reduced_embeddings, columns=[f"PCA_{i+1}" for i in range(target_dim)])
        
        logging.info(f"PCA applied. New DataFrame shape: {new_df.shape}")
        
        return new_df


    #TODO move it to separated function 
    def plot_list_images_as_matrix(self, lines=3, columns=2, imgs = []):                
        # Create a figure with high resolution
        fig, axes = plt.subplots(lines, columns, figsize=(40, 20), dpi=100)  # Adjust figsize as needed

        # Display images in high quality
        for ax, img_buff in zip(axes.flatten(), imgs):
            img = Image.open(img_buff)
            ax.imshow(img)
            ax.axis('off')  # Turn off axis for a cleaner look

        # Adjust the spacing between images        
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust space between images
        plt.tight_layout()
        plt.show()

    def plot_average_similarities_result(self, title, average_distances: dict):        
        df = pd.DataFrame(list(average_distances.items()), columns=['Key', 'Average Distance'])
        df.index = range(1, len(df) + 1)  # Set index to start from 1
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(8, len(df) * 0.5))
        # Hide axes
        ax.axis('off')

        # Create the table, including the index as a separate column
        table = ax.table(cellText=df.reset_index().values, colLabels=["Index", "Key", "Average Distance"],
                        cellLoc='left', loc='center')
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns) + 1)))  # +1 for the index column
        table.scale(1.2, 1.2)  # Adjust scale for readability
        plt.title(title, fontsize=14)
        plt.show()    
    
    # TODO add t-SNE      
    #######
    # Plot UMAP 2D
    #######
    def plot_umap_2d(self, df, centroids, title=str(), x_lim=(-25, 25), y_lim=(-25, 25)):
        """
        Plot UMAP 2D visualization with centroids marked as red dots and return the plot as an image buffer.
        
        Args:
            df (pd.DataFrame): DataFrame with UMAP coordinates for each set.
            centroids (pd.DataFrame): DataFrame with centroids for each group.
            title (str): Title of the plot.
            x_lim (tuple): Static x-axis limits.
            y_lim (tuple): Static y-axis limits.
        
        Returns:
            io.BytesIO: In-memory image buffer containing the plot.
        """
        # Create an in-memory buffer to store the image
        buf = io.BytesIO()
        
        plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

        # Get the number of columns in the DataFrame
        num_columns = df.shape[1]
        column_names = df.columns

        # Create a colormap with distinct colors based on the number of columns
        cmap = plt.cm.get_cmap('tab10', num_columns)

        # Define different markers for the objects
        markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'H', 'v', '<']  # Add more if needed
        num_markers = len(markers)

        # Plot each column's coordinates
        for i, column in enumerate(column_names):
            # Extract the UMAP coordinates for the current column
            umap_coords = np.vstack(df[column].values)
            
            # Choose marker; wrap around if more columns than markers
            marker = markers[i % num_markers]
            
            # Scatter plot for the current column with unique color and marker
            plt.scatter(umap_coords[:, 0], umap_coords[:, 1], label=column, alpha=0.7, s=20, color=cmap(i), marker=marker)
            
            # Plot the centroid as a bold red dot
            centroid = centroids.loc[column].values  # Extract centroid coordinates from DataFrame
            plt.scatter(centroid[0], centroid[1], color='red', s=50, edgecolor='black', marker='o', zorder=1)

        # Set static x and y axis limits
        plt.xlim(x_lim)
        plt.ylim(y_lim)

        # Add title, labels, and legend
        plt.title(title, fontsize=14)
        plt.xlabel("UMAP Component 1", fontsize=8)
        plt.ylabel("UMAP Component 2", fontsize=8)
        plt.legend(title="Objects", bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot

        # Automatically adjust the layout
        plt.tight_layout()
        
        # Save the plot to the in-memory buffer
        plt.savefig(buf, format='png')
        plt.close()  # Close the plot to free memory

        # Move the buffer cursor to the beginning
        buf.seek(0)

        return buf

    def plot_umap_3d(self, df, title=str(), save_path=None):
        fig = plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
        ax = fig.add_subplot(111, projection='3d')  # Set up a 3D plot

        # Get the number of columns in the DataFrame
        num_columns = df.shape[1]
        column_names = df.columns

        # Create a colormap with distinct colors based on the number of columns
        cmap = plt.cm.get_cmap('tab10', num_columns)

        # Define different markers for the objects
        markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'H', 'v', '<']
        num_markers = len(markers)

        # Plot each column's coordinates
        for i, column in enumerate(column_names):
            # Extract the UMAP coordinates for the current column
            umap_coords = np.array(df[column].tolist())
            
            # Choose marker; wrap around if more columns than markers
            marker = markers[i % num_markers]
            
            # Scatter plot for the current column with unique color and marker
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1], umap_coords[:, 2], label=column, alpha=0.7, s=20, color=cmap(i), marker=marker)

        # Add title, labels, and legend
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("UMAP Component 1", fontsize=8)
        ax.set_ylabel("UMAP Component 2", fontsize=8)
        ax.set_zlabel("UMAP Component 3", fontsize=8)
        ax.legend(title="Objects", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Automatically adjust the layout
        plt.tight_layout()
        # Save the plot to the specified path
        plt.savefig(save_path)
        # Show the plot
        plt.show()
    

    def plot_umap_12d_4x3d(self, df, title=str(), save_path=None):
        # Define sets of components to plot in 3D (four sets of three components each)
        component_sets = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
        
        # Iterate over each set of components and create a separate 3D plot
        for idx, (comp_x, comp_y, comp_z) in enumerate(component_sets):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Get the number of columns in the DataFrame
            num_columns = df.shape[1]
            column_names = df.columns

            # Create a colormap with distinct colors based on the number of columns
            cmap = plt.cm.get_cmap('tab10', num_columns)

            # Define different markers for the objects
            markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'H', 'v', '<']
            num_markers = len(markers)

            # Plot each column's coordinates for the current set of components
            for i, column in enumerate(column_names):
                # Extract the UMAP coordinates for the current column
                umap_coords = np.array(df[column].tolist())
                
                # Choose marker; wrap around if more columns than markers
                marker = markers[i % num_markers]
                
                # Scatter plot for the current column with unique color and marker
                ax.scatter(
                    umap_coords[:, comp_x], umap_coords[:, comp_y], umap_coords[:, comp_z], 
                    label=column, alpha=0.7, s=20, color=cmap(i), marker=marker
                )

            # Set plot labels and title, indicating the components used in this plot
            ax.set_title(f"{title} (Components {comp_x+1}, {comp_y+1}, {comp_z+1})", fontsize=14)
            ax.set_xlabel(f"UMAP Component {comp_x+1}", fontsize=8)
            ax.set_ylabel(f"UMAP Component {comp_y+1}", fontsize=8)
            ax.set_zlabel(f"UMAP Component {comp_z+1}", fontsize=8)
            ax.legend(title="Objects", bbox_to_anchor=(1.05, 1), loc='upper left')

            # Automatically adjust the layout
            plt.tight_layout()
            
            # Save each plot with a different name, indicating which components are plotted
            if save_path:
                plt.savefig(f"{save_path.rstrip('.png')}_comp_{comp_x+1}_{comp_y+1}_{comp_z+1}.png")
            
            # Show each plot
            plt.show()