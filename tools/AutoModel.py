import tensorflow.keras.applications as keras_applications
from tensorflow.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import os


class AutoModel(object):
    def __init__(
            self, train_path,
            valid_path=None,
            generate_image=True,
            image_shape=(100, 100, 3),
            num_of_trainable_layer=4,
            batch_size=32,
            model_function=keras_applications.vgg16.VGG16
    ):
        self.train_path = train_path
        self.valid_path = valid_path
        self.image_shape = image_shape
        self.generate_image = generate_image
        self.batch_size = batch_size
        self.history = None
        self.model = None
        self.multi_model = None
        self.train_data_gen = None
        self.train_generator = None
        self.valid_data_gen = None
        self.valid_generator = None

        self.generator_params = dict(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=0.2,
        )

        model = model_function(
            include_top=False,
            weights="imagenet",
            input_shape=self.image_shape
        )

        for layer in model.layers[:-num_of_trainable_layer]:
            layer.trainable = False

        if self.generate_image:
            self.train_data_gen = ImageDataGenerator(**self.generator_params)
        else:
            self.train_data_gen = ImageDataGenerator()

        self.train_generator = self.train_data_gen.flow_from_directory(
            self.train_path,
            target_size=self.image_shape[:2],
            batch_size=self.batch_size,
            class_mode="categorical",
            subset='training',
            shuffle=True
        )

        # if self.valid_path is not None:
        # self.valid_data_gen = ImageDataGenerator(rescale=1.0/255)
        self.valid_generator = self.train_data_gen.flow_from_directory(
            self.train_path,
            target_size=self.image_shape[:2],
            batch_size=self.batch_size,
            class_mode="categorical",
            subset='validation',
            shuffle=False
        )

        input_layer = layers.Input(shape=self.image_shape, name="Input_Layer")
        x = model(input_layer)
        x = layers.Flatten()(x)
        output_layer = layers.Dense(units=self.train_generator.num_classes, activation="softmax", name="Softmax")(x)
        self.model = models.Model(input_layer, output_layer)
        self.model.summary()

    def train(self, epochs=10, fine_tune=0, transfer_lr=1e-5, fine_tune_lr=1e-6, save_path=None):
        train_steps = self.train_generator.samples // self.train_generator.batch_size
        valid_steps = None
        if self.valid_generator is not None:
            valid_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.compile(
            loss=losses.categorical_crossentropy,
            optimizer=optimizers.Adam(lr=transfer_lr),
            metrics=["accuracy"]
        )

        history1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=train_steps,
            initial_epoch=0,
            epochs=epochs,
            validation_data=self.valid_generator,
            validation_steps=valid_steps,
            verbose=1
        )

        for layer in self.model.layers:
            layer.trainable = True

        self.model.compile(
            loss=losses.categorical_crossentropy,
            optimizer=optimizers.RMSprop(lr=fine_tune),
            metrics=["accuracy"]
        )

        history2 = self.model.fit(
            self.train_generator,
            steps_per_epoch=train_steps,
            initial_epoch=epochs,
            epochs=epochs + fine_tune,
            validation_data=self.valid_generator,
            validation_steps=valid_steps,
            verbose=1
        )

        self.history = history1.history
        for key, val in self.history.items():
            self.history[key] += history2.history[key]

        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))

            self.model.save(save_path)

            name = ".".join(save_path.split(".")[:-1])
            f = open(name + ".txt", "w")
            for label in self.train_generator.class_indices:
                f.write(label + "\n")
            f.close()

            plot_model(self.model, name + ".png", show_shapes=True, show_layer_names=True)

    def show_history(self, accuracy_path=None, loss_path=None):
        history = self.history

        if "accuracy" in history:
            acc = history["accuracy"]
        else:
            acc = history["acc"]
        loss = history["loss"]
        epochs = range(len(acc))

        plt.figure()
        plt.plot(epochs, acc, "b", label="Training acc")
        if "val_accuracy" in history:
            plt.plot(epochs, history["val_accuracy"], "r", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.legend()

        if accuracy_path is not None:
            if not os.path.exists(os.path.dirname(accuracy_path)):
                os.mkdir(os.path.dirname(accuracy_path))
            plt.savefig(accuracy_path)

        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        if "val_loss" in history:
            plt.plot(epochs, history["val_loss"], "r", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()

        if loss_path is not None:
            if not os.path.exists(os.path.dirname(loss_path)):
                os.mkdir(os.path.dirname(loss_path))
            plt.savefig(loss_path)
        plt.show()


if __name__ == "__main__":
    _model = AutoModel("../dataset/train/")
    _model.train(epochs=50, fine_tune=50, save_path="../model_data/model.h5")
    _model.show_history("../model_data/history/acc.png", "../model_data/history/loss.png")
