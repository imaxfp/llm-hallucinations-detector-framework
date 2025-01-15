import io
import logging
import matplotlib.pyplot as plt  # You still need matplotlib as a backend
import numpy as np
import pandas as pd
import warnings
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

    #TODO move me to the separated Embedding Service
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
    
    def find_centroids(self, df):        
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
    
    
    def apply_umap_generate_list_2d_imgs(self, amount_imgs=1):
        imgs = []
        df_umap = self.apply_umap(n_components=2, rand_state=42)
        centroids_df = pd.DataFrame(columns=df_umap.columns)
        for i in range(amount_imgs):    
            df_umap = self.apply_umap(n_components=2, rand_state=i)
            centroids_df = self.find_centroids(df_umap)
            img_buff = self.plot_umap_2d(df=df_umap, 
                 centroids=centroids_df, 
                 title=f"True Answers VS LLMs answers and Hallucinations. Shape = {df_umap.shape}")
            imgs.append(img_buff)    
        return imgs    
    
    #TODO hold me in the Visual Service
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
    def plot_umap_2d(self, df, centroids, title=str(), x_lim=(-15, 15), y_lim=(-15, 15)):
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

        # Center the x_lim and y_lim around the mean position of all data
        all_coords = np.vstack([np.vstack(df[col].values) for col in df.columns])
        center_x = float(np.mean(all_coords[:, 0]))
        center_y = float(np.mean(all_coords[:, 1]))
        x_lim = (center_x + float(x_lim[0]), center_x + float(x_lim[1]))
        y_lim = (center_y + float(y_lim[0]), center_y + float(y_lim[1]))

        # Create a colormap with distinct colors based on the number of columns
        cmap = plt.cm.get_cmap('tab10', num_columns)

        # Define different markers for the objects
        markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'H', 'v', '<']
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

        # Set calculated x and y axis limits
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