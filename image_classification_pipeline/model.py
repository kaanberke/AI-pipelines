# %% LIBRARIES
import datetime
import os
import uuid
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

import logging


# %% SETTINGS
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except IndexError as e:
    logging.error("Physical devices " + str(e))

# %% CUSTOM CALLBACK(S)
class PerformanceVisualizationCallback(Callback):
    def __init__(self,
                 model: tf.keras.Model,
                 model_no: int,
                 test_data: tf.Tensor):
        super().__init__()
        self.model = model
        self.model_no = model_no
        self.test_data = test_data
        self.csv_path = f"./results/{self.model_no}_results.csv"

        if not os.path.isdir("./results"):
            os.mkdir("results")

    def on_epoch_end(self,
                     epoch: int,
                     logs: dict = None):
        if logs is None:
            logs = {}
        loss, acc, tp, tn, fn, fp = self.model.evaluate(self.test_data)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "a") as f:
                f.write("Epoch,loss,acc,TN,FP,FN,TP\n")
        with open(self.csv_path, "a") as f:
            f.write(f"{epoch},{loss},{acc},{tn},{fp},{fn},{tp}\n")
        print("#" * 200)


# %% MODEL CLASS
class Model(object):

    def __init__(self,
                 input_shape: list[int],
                 class_mode: str = "binary",
                 num_classes: int = 1,
                 batch_size: int = 16,
                 epochs: int = 50,
                 learning_rate: float = 1e-4):
        """

        :param input_shape: Input shape for the model.
        :param class_mode: Desired class mode.
        :param batch_size: How many batches that is wanted.
        :param epochs: How long the model will be trained.
        :param learning_rate: Learning rate for the optimizer.

        >>> model = Model(
        ...     input_shape=[240, 240, 3],
        ...     class_mode="binary",
        ...     batch_size=16,
        ...     epochs=50,
        ...     learning_rate=1e-4
        ... )
        """
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.class_mode = class_mode
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.output_activation = {
            "binary": "sigmoid",
            "categorical": "softmax"
        }

        self.model = self.__build_model()

    def __build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.applications.ResNet50V2(include_top=False,
                                             input_shape=self.input_shape,
                                             weights="imagenet")(inputs)

        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)

        x = tf.keras.layers.Dense(1024, activation="relu", name="top_dense_0")(x)
        x = tf.keras.layers.Dropout(0.5, name="top_dropout_0")(x)

        x = tf.keras.layers.Dense(1024, activation="relu", name="top_dense_1")(x)
        x = tf.keras.layers.Dropout(0.5, name="top_dropout_1")(x)

        outputs = tf.keras.layers.Dense(self.num_classes,
                                        activation=self.output_activation[self.class_mode],
                                        name="pred")(x)

        # Compile
        _model = tf.keras.Model(inputs, outputs, name="ResNet50V2")

        optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate,
                                                decay=self.learning_rate / self.epochs)

        _model.compile(
            optimizer=optimizer,
            loss=f"{self.class_mode}_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.TruePositives(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.FalseNegatives(),
                tf.keras.metrics.FalsePositives(),
            ])

        print(_model.summary())
        return _model

    def fit(self, *args, **kwargs):
        pprint(kwargs)
        kwargs["batch_size"] = self.batch_size if "batch_size" not in kwargs else kwargs.get("batch_size")
        kwargs["epochs"] = self.epochs if "epochs" not in kwargs else kwargs.get("epochs")
        self.model.fit(*args, **kwargs)
