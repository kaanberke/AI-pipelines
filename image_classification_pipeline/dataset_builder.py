# %% LIBRARIES
import glob
import os
import random
from pathlib import Path
from typing import Union

import cv2
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.python.data import AUTOTUNE
from tqdm import tqdm


# %% Class


class Dataset(object):
    def __init__(self,
                 data_directories: list[list[str, int]],
                 labels: list,
                 image_size: tuple[int, int],
                 channels: int,
                 batch_size: int,
                 image_extensions: list[str]):
        """

        :param data_directories: Data directories should be provided
        with the number of image that is requested to be collected. -1 = all
        :param labels: Labels for the provided data directories.
        :param image_size: Image size in order to resize all
        the images for the input of the model.
        :param channels: Number of channels of the image
        :param image_extensions: Extensions of images that is
        requested to be collected
        :param batch_size: Batch size for tf.data

        >>> d = Dataset(
        ...     data_directories=[
        ...         ["./data/train/real/1", -1],
        ...         ["./data/train/real/2", -1],
        ...         ["./data/train/fake/1", -1],
        ...         ["./data/train/fake/2", -1],
        ...     ],
        ...     labels=["real", "fake"],
        ...     image_size=(240, 240),
        ...     channels=3,
        ...     batch_size=8,
        ...     image_extensions=["/*.jpeg", "/*.jpg", "/*.png"]
        ... )
        """

        super(Dataset, self).__init__()
        self.data_directories = data_directories
        self.labels = labels
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.image_extensions = image_extensions + [ext.swapcase() for ext in image_extensions]

    def get_image_paths(self):
        all_data = []
        progress_bar = tqdm(self.data_directories, desc="data collecting")
        for data_dir, sample_no in self.data_directories:
            files_grabbed = []
            for ext in self.image_extensions:
                files_grabbed.extend(glob.glob(str(data_dir) + ext))

            random.shuffle(files_grabbed)
            if sample_no == -1 or len(files_grabbed) < sample_no:
                all_data.extend(files_grabbed)
            else:
                all_data.extend(files_grabbed[:sample_no])
            progress_bar.update()
        return all_data

    def display_image_grid(self,
                           images_filepaths: list[str],
                           image_size: list[int] = (240, 240),
                           predicted_labels: list[str] = None,
                           cols: int = 5):
        """
        :param images_filepaths: File paths of the images that wants be visualized.
        :param predicted_labels: Predicted labels of the given images (optional).
        :param image_size: Image size in order to resize all the given images.
        :param cols: How many columns will be on the figure.
        :return: None
        """
        rows = len(images_filepaths) // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
        for i, image_filepath in enumerate(images_filepaths):
            image = cv2.imread(image_filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, image_size)
            true_label = list(
                set(self.labels).intersection(
                    os.path.normpath(image_filepath).split(os.sep)
                )
            )[0]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_title(true_label)
            if predicted_labels:
                predicted_label = predicted_labels[i] if predicted_labels else true_label
                color = "green" if true_label == predicted_label else "red"
                ax.ravel()[i].set_title(predicted_label, color=color)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()

    def __load_images(self,
                      image_path: tf.Tensor,
                      image_label: tf.Tensor,
                      categorical: bool = False):

        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=self.channels)
        image = tf.image.resize(image, self.image_size) / 255.0
        if categorical:
            image_label = tf.one_hot(tf.strings.to_number(image_label, out_type=tf.dtypes.int32), len(self.labels))
        return image, image_label

    @tf.function
    def __augment_images(self,
                         images: tf.Tensor,
                         labels: tf.Tensor):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.rot90(images)
        images = tf.image.random_flip_up_down(images)
        return images, labels

    def get_dataset(
            self,
            split_ratios: Union[
                tuple[float, float, float],
                tuple[float, float]] = (0.7, 0.2, 0.1),
            cache: str = "./",
            balanced_weights: bool = False,
            categorical: bool = True,
            show_details: bool = True):
        """
        :param split_ratios: (train, validation, test) ratios.
        :param cache: Path that is cached values will be placed.
        :param balanced_weights: If True, Scikit-learn's class_weight function will be applied.
        :param categorical: If True labels will be returned as categorical, else binary.
        :param show_details: If True information will be printed occasionally.
        :return: (train_dataset, validation_dataset)
                + test_dataset if split_ratios length is 3.
                + class_weights if balanced_weights is True.
        """

        cache_path = Path(cache)
        result = []
        all_data = self.get_image_paths()
        df = pd.DataFrame(all_data, columns=["image_path"])
        df["label"] = None

        # iterate over all the labels and substitute labels with relevant integer
        # value where the path involves the keyword (case insensitive).
        for idx, label in enumerate(self.labels):
            df.loc[(df["image_path"].str.contains(label, case=False)), "label"] = str(idx)

        # Show the frequency of the labels
        if show_details:
            print(df["label"].value_counts())
            print(df.tail())

        train_df, validation_df = train_test_split(df.sample(frac=1),
                                                   test_size=split_ratios[1])

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_df["image_path"].values, train_df["label"].values))

        train_dataset = (train_dataset
                         .shuffle(len(train_df))
                         .map(
                             lambda image_path, image_label: self.__load_images(image_path, label, categorical),
                             num_parallel_calls=AUTOTUNE)
                         .batch(self.batch_size)
                         .cache(str(cache_path / "train_dataset"))
                         .prefetch(AUTOTUNE))
        result.append(train_dataset)

        if len(split_ratios) == 3:
            test_size = split_ratios[2] / (split_ratios[1] + split_ratios[2])
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (validation_df["image_path"].values, validation_df["label"].values))
            validation_dataset = (validation_dataset
                                  .shuffle(len(validation_df))
                                  .map(
                                      lambda images, labels: self.__load_images(images, labels, categorical),
                                      num_parallel_calls=AUTOTUNE, )
                                  .batch(self.batch_size)
                                  .cache(str(cache_path / "validation_dataset"))
                                  .prefetch(AUTOTUNE))

            result.append(validation_dataset)

            validation_df, test_df = train_test_split(validation_df.sample(frac=1),
                                                      test_size=test_size)

            test_dataset = tf.data.Dataset.from_tensor_slices(
                (test_df["image_path"].values, test_df["label"].values))
            test_dataset = (test_dataset
                            .shuffle(len(test_df))
                            .map(
                                lambda images, labels: self.__load_images(images, labels, categorical),
                                num_parallel_calls=AUTOTUNE, )
                            .batch(1)
                            .cache(str(cache_path / "test_dataset"))
                            .prefetch(AUTOTUNE))
            result.append(test_dataset)
        elif len(split_ratios) == 2:
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (validation_df["image_path"].values, validation_df["label"].values))

            validation_dataset = (validation_dataset
                                  .shuffle(len(validation_df))
                                  .map(
                                      lambda images, labels: self.__load_images(images, labels, categorical),
                                      num_parallel_calls=AUTOTUNE, )
                                  .batch(self.batch_size)
                                  .cache(str(cache_path / "validation_dataset"))
                                  .prefetch(AUTOTUNE))

            result.append(validation_dataset)
        else:
            raise ValueError("Improper value for split_ratios parameter")

        if balanced_weights:
            class_weights = class_weight.compute_class_weight(
                "balanced", classes=df["label"].unique(), y=df["label"])
            class_weights = dict(enumerate(class_weights))
            print(f"{class_weights=}")
            result.append(class_weights)

        return result
