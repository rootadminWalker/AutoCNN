from tensorflow.keras.applications import vgg16
import cv2
import numpy as np
import os


model = vgg16.VGG16(weights="imagenet", include_top=True)
directory = "../datasets/dogs-vs-cats/test1/"
for name in os.listdir(directory):
    path = os.path.join(directory, name)

    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.reshape(1, 224, 224, 3)
    image = image.astype(np.float32)

    predicts = model.predict(image)

    idx = vgg16.decode_predictions(predicts, top=3)[0]

    print(idx)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)