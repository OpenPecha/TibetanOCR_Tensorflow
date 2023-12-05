"""
A selective collection of CTC-based network architectures to allow easy experimentation.

"""


import keras
import logging
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import (
    Add,
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GRU,
    Input,
    Lambda,
    LSTM,
    MaxPooling2D,
    Multiply,
    RandomRotation,
)

from keras.optimizers import Adam, RMSprop, Lion


# optional: define a stack of augmentations
# see here for examples: https://www.tensorflow.org/tutorials/images/data_augmentation
#
data_augmentation = keras.Sequential(
    [
        RandomRotation(factor=(0.01, 0.02)),
    ]
)
"""
1) Easter2: https://github.com/kartikgill/Easter2/blob/main/src/easter_model.py
# uses standard values from the reference implementation

"""

BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.997


def ctc_loss(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_custom(args):
    """
    custom CTC loss, see Easter2 paper
    """
    y_pred, labels, input_length, label_length = args
    ctc_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    p = tf.exp(-ctc_loss)
    gamma = 0.5
    alpha = 0.25
    return alpha * (K.pow((1 - p), gamma)) * ctc_loss


def batch_norm(inputs):
    return BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(
        inputs
    )


def add_global_context(data, filters):
    """
    1D Squeeze and Excitation Layer.
    """
    pool = GlobalAveragePooling1D()(data)

    pool = Dense(filters // 8, activation="relu")(pool)

    pool = Dense(filters, activation="sigmoid")(pool)

    final = Multiply()([data, pool])
    return final


def easter_unit(old, data, filters, kernel, stride, dropouts):
    """
    Easter unit with dense residual connections
    """
    old = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same")(old)
    old = batch_norm(old)

    this = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same")(data)
    this = batch_norm(this)

    old = Add()([old, this])

    data = Conv1D(filters=filters, kernel_size=kernel, strides=stride, padding="same")(
        data
    )

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(dropouts)(data)

    data = Conv1D(filters=filters, kernel_size=kernel, strides=stride, padding="same")(
        data
    )

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(dropouts)(data)

    data = Conv1D(filters=filters, kernel_size=kernel, strides=stride, padding="same")(
        data
    )

    data = batch_norm(data)
    data = add_global_context(data, filters)

    final = Add()([old, data])

    data = Activation("relu")(final)
    data = Dropout(dropouts)(data)

    return data, old


def Easter2(
    input_width: int = 2000,
    input_height: int = 80,
    classes: int = 68,
    learning_rate: float = 0.001,
    optimizer: str = "Adam",
    output_dim: int = 500,  # standard value in paper, experiment with other values if necessary
):
    inputs = Input(name="images", shape=(input_width, input_height))

    # data = data_augmentation(inputs) # optional: use augmentation layers, some keras layers cause problems though
    data = Conv1D(filters=128, kernel_size=3, strides=2, padding="same")(inputs)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)

    data = Conv1D(filters=128, kernel_size=3, strides=2, padding="same")(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)

    old = data

    # 3 * 3 Easter Blocks (with dense residuals)
    data, old = easter_unit(old, data, 128, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 7, 1, 0.2)
    data, old = easter_unit(old, data, 256, 9, 1, 0.3)

    data = Conv1D(
        filters=512, kernel_size=11, strides=1, padding="same", dilation_rate=2
    )(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.4)(data)

    data = Conv1D(filters=512, kernel_size=1, strides=1, padding="same")(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.4)(data)

    data = Conv1D(filters=classes, kernel_size=1, strides=1, padding="same")(data)

    y_pred = Activation("softmax", name="Final")(data)

    """
       define an optimizer, check the docs for optional parameters
       - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
       - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/RMSprop
       - TODO: parameterize optimizer parameters, add Lion optimizer for tf > 2.12?
       """

    if optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == "RMSProp":
        optimizer = RMSprop(learning_rate=learning_rate, centered=True)
    elif optimizer == "Lion":
        optimizer = Lion(learning_rate=learning_rate)
    else:
        logging.info(f"setting optimzer to default: Adam")
        optimizer = Adam(learning_rate=learning_rate)

    labels = Input(name="labels", shape=[output_dim], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    output = Lambda(ctc_custom, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    # compiling model
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=output)
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    return model
