from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os


model = load_model("model_data/model.h5")
label = []
f = open("model_data/model.txt")
for line in f.readlines():
    label.append(line.strip())
f.close()

directory = "dataset/test"
for name in os.listdir(directory):
    path = os.path.join(directory, name)

    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))
    image = image.reshape(1, 100, 100, 3)
    image = image.astype(np.float32)
    image = image / 255

    predicts = model.predict(image)

    idx = np.argmax(predicts)

    print(idx, label[idx])
    cv2.putText(frame, label [idx], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
