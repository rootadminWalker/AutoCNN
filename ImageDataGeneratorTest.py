from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import cv2 as cv
import numpy as np

origin = load_img('./images/sample.jpg')
origin = img_to_array(origin)
origin = np.array([origin])

generator = ImageDataGenerator()

for gen_image in generator.flow(origin):
    gen_image = cv.cvtColor(gen_image[0], cv.COLOR_RGB2BGR)
    gen_image = gen_image.astype(np.uint8)
    cv.imshow('frame', gen_image)
    cv.waitKey(0)
