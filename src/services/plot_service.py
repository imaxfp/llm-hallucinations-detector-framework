import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Visualization function for scatter plot
@DeprecationWarning
def visualize_similarity_3d_scatter(question_entities):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')

    # Colors and markers for each model
    markers = ['o', 'v', '^', 's', 'p']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Extract model names from the first entity (assuming all have same models)
    model_names = list(question_entities[0].long_answer_similarity.keys())
    
    for idx, question_entity in enumerate(question_entities):
        # Get similarity values
        similarities = list(question_entity.long_answer_similarity.values())
        
        # X-axis: index of models, Y-axis: different question entities, Z-axis: similarity values
        x = np.arange(len(model_names))  # X-axis corresponds to the model index
        y = np.full_like(x, idx, dtype=float)  # Y-axis corresponds to the question entity index
        z = similarities  # Z-axis is the similarity value
        
        # Plot each model in 3D scatter with different colors and markers
        for i, model in enumerate(model_names):
            ax.scatter3D(x[i], y[i], z[i], color=colors[i % len(colors)], marker=markers[i % len(markers)], s=100)
    
    # Set labels
    ax.set_xlabel('Model Index')
    ax.set_ylabel('Question Entity Index')
    ax.set_zlabel('Similarity')

    # Set xticks as model names
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names)

    # Title
    ax.set_title('3D Scatter Plot of Long Answer Similarity (Clustered by Models)')
    
    plt.show()
