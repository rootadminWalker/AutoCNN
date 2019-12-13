from tools import AutoModel

model = AutoModel("../datasets/dogs_vs_cats/train/")
model.train(110, "model_data/model.h5")
model.show_history("model_data/acc.png", "model_data/loss.png")
