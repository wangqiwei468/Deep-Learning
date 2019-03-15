from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')
model.summary()     # 作为提醒

#%% 预处理单张图像
img_path = 'E:/Python_Process/deep_learning_with_python/5_deep_learning_to_machine_vision/cats_and_dogs_small/test/cats/cat.1700.jpg'

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor /= 255.

# 其形状为（1, 150, 150, 3）
print(img_tensor.shape)

#%% 显示测试图像
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

#%% 用一个输入张量和一个输出张量列表将模型实例化
from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]    # 提取前八层的输出
activation_model = models.Model(inputs = model.input, outputs = layer_outputs) # 创建一个模型，给定模型的输入输出

#%% 以预测模式运行模型
activations = activation_model.predict(img_tensor)  # 返回8个Numpy数组组成的列表，每个层激活对应一个Numpy数组

#%% 将第1层第4个通道可视化
import matplotlib.pyplot as plt
first_layer_activations = activations[0]
plt.matshow(first_layer_activations[0, :, :, 4], cmap='viridis')
plt.show()

#%%
plt.matshow(first_layer_activations[0, :, :, 7], cmap='viridis')
plt.show()

#%% 将每个中间激活的所有通道可视化
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features //  images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col*images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                        row * size : (row + 1) * size] = channel_image

    scale = 1./size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap = 'viridis')
    plt.savefig("E:/Python_Process/deep_learning_with_python/5_deep_learning_to_machine_vision/5.4_Visualization_Intermediate_Activation/%s.jpg"%(layer_name))
    plt.show()
