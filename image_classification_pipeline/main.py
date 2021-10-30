import os
from datetime import datetime

import tensorflow as tf

from dataset_builder import Dataset
from model import Model, PerformanceVisualizationCallback

d = Dataset(
    data_directories=[
        ["./data/train/real/1", -1],
        ["./data/train/real/2", -1],
        ["./data/train/fake/1", -1],
        ["./data/train/fake/2", -1],

        ["./data/validation/real/1", -1],
        ["./data/validation/real/2", -1],
        ["./data/validation/fake/1", -1],
        ["./data/validation/fake/2", -1],
    ],
    labels=["real", "fake"],
    image_size=(240, 240),
    channels=3,
    batch_size=16,
    image_extensions=["/*.jpeg", "/*.jpg", "/*.png"]
)

train_dataset, val_dataset, test_dataset, class_weights = d.get_dataset(
    split_ratios=(0.7, 0.2, 0.1),
    cache=".",
    balanced_weights=True,
    categorical=True,
    show_details=True
)

m = Model(
    input_shape=[240, 240, 3],
    class_mode="categorical",
    num_classes=2,
    batch_size=16,
    epochs=50,
    learning_rate=1e-4
)

initial_epoch = 0
model_no = 9999
load_model_answer = input(
    "Would you like to load model from existing checkpoint? (y/n) ")
if load_model_answer.lower() == "y":
    checkpoint_files = os.listdir("checkpoint")
    checkpoint_files = sorted(checkpoint_files)
    for i in range(len(checkpoint_files)):
        if checkpoint_files[i].endswith(".h5"):
            print(i, checkpoint_files[i])
    try:
        checkpoint_number = int(
            input(
                "Please enter the number of the checkpoint you want to load: ")
        )
        m.model.load_weights("./checkpoint_id/" +
                             checkpoint_files[checkpoint_number])
        model_no, epoch_no, _ = checkpoint_files[checkpoint_number].split("_")
        initial_epoch = int(epoch_no)
    except ValueError:
        print(
            "You've entered an invalid number. Training is being started from scratch.."
        )
else:
    model_no = input("Model no:")

checkpoint_filepath = "checkpoint"
if not os.path.exists(checkpoint_filepath):
    os.mkdir(checkpoint_filepath)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath, model_no + "_{epoch:02d}_{val_loss:.4f}.h5"),
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=False)

performance_callback = PerformanceVisualizationCallback(
    model=m.model,
    model_no=model_no,
    test_data=test_dataset,
)

log_dir = os.path.join("logs", "fit", datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "_model_" + model_no)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)
# %tensorboard --logdir ./logs/fit

history = m.fit(
    train_dataset,
    validation_data=val_dataset,
    verbose=1,
    callbacks=[
        tensorboard_callback,
        model_checkpoint_callback,
        performance_callback
    ],
    class_weight=class_weights)
