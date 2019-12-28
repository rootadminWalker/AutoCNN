from tools import AutoModel

model = AutoModel("../datasets/dogs-vs-cats/cats_and_dogs_small/train/")
model.train(10, "model_data/model.h5")
model.show_history("model_data/acc.png", "model_data/loss.png")
