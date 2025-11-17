import numpy as np
import matplotlib.pyplot as plt

# loaded_array = np.load('data/full_numpy_bitmap_triangle.npy')


# num_amostras = 2000
# total_imagens = loaded_array.shape[0]

# indices = np.random.choice(total_imagens, size=num_amostras, replace=False)

# subset = loaded_array[indices]

# print(subset.shape)   # (2000, 784)

# np.save('data/triangle_subset_2000.npy', subset)

circle_ds = np.load('data/circle_subset_2000.npy')
square_ds = np.load('data/square_subset_2000.npy')
triangle_ds = np.load('data/triangle_subset_2000.npy')

circle_ds = circle_ds.reshape((-1, 28, 28))
square_ds = square_ds.reshape((-1, 28, 28))
triangle_ds = triangle_ds.reshape((-1, 28, 28))

print(circle_ds.shape)
print(square_ds.shape)
print(triangle_ds.shape)

# fig, axs = plt.subplots(3, 3, figsize=(9, 9))

# # 3 círculos
# for i in range(3):
#     axs[0, i].imshow(circle_ds[i], cmap='gray')
#     axs[0, i].set_title(f'Circle {i}')
#     axs[0, i].axis('off')

# # 3 quadrados
# for i in range(3):
#     axs[1, i].imshow(square_ds[i], cmap='gray')
#     axs[1, i].set_title(f'Square {i}')
#     axs[1, i].axis('off')

# # 3 triângulos
# for i in range(3):
#     axs[2, i].imshow(triangle_ds[i], cmap='gray')
#     axs[2, i].set_title(f'Triangle {i}')
#     axs[2, i].axis('off')

# plt.tight_layout()
# plt.show()

# juntar as imagens
X = np.concatenate([circle_ds, square_ds, triangle_ds], axis=0)

# criar os rótulos
y_circle   = np.zeros(circle_ds.shape[0], dtype=int)   # classe 0
y_square   = np.ones(square_ds.shape[0], dtype=int)    # classe 1
y_triangle = np.full(triangle_ds.shape[0], 2, dtype=int)  # classe 2

y = np.concatenate([y_circle, y_square, y_triangle], axis=0)

print(X.shape)  # (6000, 28, 28)
print(y.shape)  # (6000,)

print("Antes da normalização:", X.min(), X.max())  # deve ser algo como 0 255
X = X.astype("float32") / 255.0

print("Depois da normalização:", X.min(), X.max())  # deve ser algo como 0.0 1.0

X = X.reshape(-1, 28, 28, 1)

print("Novo shape de X:", X.shape)
