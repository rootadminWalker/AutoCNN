from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os


model = load_model("model_data/model.h5")
f = open("model_data/model.txt")
labels = f.readlines()
f.close()

directory = "../datasets/dogs-vs-cats/test1/"
for name in os.listdir(directory):
    path = os.path.join(directory, name)

    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(frame, (100, 100))
    image = image.reshape(1, 100, 100, 3)
    image = image.astype(np.float32)
    image = image / 255

    predicts = model.predict(image)

    idx = np.argmax(predicts)

    print(idx, labels[idx])
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
