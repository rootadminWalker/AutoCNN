from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

label = []
with open("/home/root_walker/workspace/AutoCNN/model_data/VGG16/model.txt") as f:
    for line in f.readlines():
        label.append(line.strip())
    f.close()

model = load_model("/home/root_walker/workspace/AutoCNN/model_data/VGG16/model.h5")

gen = ImageDataGenerator(rescale=1.0 / 255.0)
generator = gen.flow_from_directory('../datasets/dataset_pose_test')

result = model.evaluate(x=generator)
print(result)
