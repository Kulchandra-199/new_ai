# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Example: Visualize a 10x10 weight matrix
# weights = np.random.rand(10, 10)  # Replace with your model's weights

# # Create grid
# x = np.arange(weights.shape[0])
# y = np.arange(weights.shape[1])
# x, y = np.meshgrid(x, y)

# # Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, weights, cmap='viridis')
# ax.set_xlabel('X (Neuron Input)')
# ax.set_ylabel('Y (Neuron Output)')
# ax.set_zlabel('Weight Value')
# plt.title('3D Weight Matrix Surface')
# plt.show()

import satlaspretrain_models
model = satlaspretrain_models.Weights().get_pretrained_model("Sentinel2_SwinB_SI_RGB")