from tools import AutoModel

model = AutoModel("./dataset/train/")
model.train(epochs=100, fine_tune=100, save_path="./model_data/model.h5")
model.show_history("./model_data/history/acc.png", "./model_data/history/loss.png")
