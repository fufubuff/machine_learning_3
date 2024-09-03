import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

# Load the image
path = "stones.jpg"  # Adjust path as necessary
image = Image.open(path)
image_np = np.array(image)

# Reshape the image to be a list of pixels
pixels = image_np.reshape(-1, 3)

# Number of clusters
ks = range(1, 7)  # Clustering for k = 1 to 6

# Set up plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Adjust subplot grid for display
axes = axes.flatten()  # Flatten the array of axes

for i, k in enumerate(ks):
    if k == 1:
        # For k=1, display the original image
        result_image = image_np
    else:
        # Apply K-Means clustering to the pixel values for k > 1
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(pixels)
        
        # Create a new image from the labels
        new_colors = kmeans.cluster_centers_.astype(int)[labels]
        result_image = new_colors.reshape(image_np.shape)
    
    # Display results
    axes[i].imshow(result_image)
    axes[i].set_title(f'k={k}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
