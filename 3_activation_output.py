from tensorflow.keras.models import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


model = load_model("./model_data/model.h5")
f = open("./model_data/model.txt")
labels = f.readlines()
f.close()

frame = cv2.imread("./dataset/test/20191228155603.jpg")
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (100, 100))
image = image.reshape(1, 100, 100, 3)
image = image.astype(np.float32)
image = image / 255
print(image.shape)

input1 = None
layers = []
for l1 in model.layers[1:]:
    if type(l1) == Model:
        m1: Model = l1
        input1 = m1.input
        for l2 in m1.layers[1:]:
            layers.append(l2.output)

activation_model = Model(inputs=input1, outputs=layers)
activations = activation_model.predict(image)
k = 0
for activation in activations:
    _, w, h, d = activation.shape
    cols = int(np.sqrt(d))
    rows = int(np.ceil(d / cols))
    f, axarr = plt.subplots(rows, cols)
    plt.grid(False)
    plt.axis("off")
    print(k, activation.shape)

    i = 0
    for row in range(rows):
        for col in range(cols):
            axarr[row, col].set_xticks([])
            axarr[row, col].set_yticks([])

            if i >= d:
                axarr[row, col].imshow(np.zeros((100, 100)), cmap="viridis")
            else:
                img = activation[0, :, :, i]
                axarr[row, col].imshow(img, cmap="viridis")
            i += 1
    if not os.path.exists(os.path.dirname("./model_data/layer_outputs/")):
        os.mkdir(os.path.dirname("./model_data/layer_outputs/"))
    plt.savefig("./model_data/layer_outputs/%d.png" % k)
    k += 1
    # plt.show()
