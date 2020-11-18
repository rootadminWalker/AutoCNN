import os

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet import ResNet50, ResNet152
from tensorflow.keras.applications.vgg19 import VGG19
from tools.AutoModel import AutoModel

model_list = [
    # VGG16,
    # VGG19,
    # MobileNet,
    # MobileNetV2,
    ResNet50,
    # ResNet152,
    # InceptionV3,
    # InceptionResNetV2,
]

for model_function in model_list:
    save_dir = f'./model_data/{model_function._keras_api_names[0].split(".")[-1]}'
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    model = AutoModel("../datasets/dataset_pose", model_function=model_function)
    model.train(epochs=5, fine_tune=25, transfer_lr=1e-3, save_path=f"./{save_dir}/model.h5")

    model.show_history(f"./{save_dir}/acc.png", f"./{save_dir}/loss.png")
