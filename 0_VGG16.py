from tensorflow.keras.applications import vgg16
import cv2
import numpy as np


model = vgg16.VGG16(weights="imagenet")

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.reshape(1, 224, 224, 3)
    image = image.astype(np.float32)

    predicts = model.predict(image)

    ans = vgg16.decode_predictions(predicts, top=3)[0]

    print(ans)
    cv2.imshow("frame", frame)

    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()
