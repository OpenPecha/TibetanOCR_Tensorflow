import os
import json
import pyewts
import logging
import numpy as np
import tensorflow as tf
from glob import glob
from typing import Optional, List
from datetime import datetime
from natsort import natsorted

from keras.models import Model
from keras.layers import StringLookup
from keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
)
from keras.utils.data_utils import Sequence
from src.Models import Easter2
from src.Utils import (
    shuffle_data,
    read_data2,
    ImageReader,
    validate_data,
    build_charset,
)

logging.getLogger().setLevel(logging.INFO)


class OCRDataset:
    """
    Handles the initialization of the dataset.
    Assumes that the line images are in directory/lines/ and the labels in directory/transcriptions.
    Change these paths if necessary.
    """

    def __init__(
        self,
        directory: str,
        train_test_split: float = 0.8,
        batch_size: int = 32,
        charset: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> None:
        self._directory = directory
        self._ds_images = []
        self._ds_labels = []
        self._train_idx = []
        self._val_idx = []
        self._test_idx = []
        self._charset = charset
        self._converter = pyewts.pyewts()
        self.batch_size = batch_size
        self._train_test_split = train_test_split
        self._time_stamp = datetime.now()
        self.output_dir = self.get_output_dir(output_dir)

        self._init()

    def _init(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        ds_images = natsorted(glob(f"{self._directory}/lines/*.jpg"))
        ds_labels = natsorted(glob(f"{self._directory}/transcriptions/*.txt"))
        logging.info(f"Total Images: {len(ds_images)}, Total Labels: {len(ds_labels)}")

        common_list = validate_data(ds_images, ds_labels)

        images = list(map(self._map_img_dir, common_list))
        labels = list(map(self._map_label_dir, common_list))

        images, labels = read_data2(images, labels, self._converter)

        self._ds_images, self._ds_labels = shuffle_data(images, labels)

        (self._train_idx, self._val_idx, self._test_idx) = self._create_sets(
            self._ds_images, self._ds_labels
        )

        logging.info(f"Train Images: {len(self._train_idx)}")
        logging.info(f"Validation Images: {len(self._val_idx)}")
        logging.info(f"Test Images: {len(self._test_idx)}")

        if self._charset is None:
            self.build_charset()

        self._save_dataset("train")
        self._save_dataset("val")
        self._save_dataset("test")


    def get_output_dir(self, output_dir):
        time_stamp = f"{self._time_stamp.year}_{self._time_stamp.month}_{self._time_stamp.day}_{self._time_stamp.hour}_{self._time_stamp.minute}"
        if output_dir is None:
            return os.path.join(self._directory, "Output", time_stamp)
        else:
            return os.path.join(output_dir, self._directory, time_stamp)

    def _map_img_dir(self, x):
        return f"{self._directory}/lines/{x}.jpg"

    def _map_label_dir(self, x):
        return f"{self._directory}/transcriptions/{x}.txt"

    def _create_sets(
        self, images: list[str], labels: list[str]
    ) -> tuple[list[int], list[int], list[int]]:
        max_batches = (len(images) - (len(images) % self.batch_size)) // self.batch_size

        train_batches = int(max_batches * self._train_test_split)
        val_batches = (max_batches - train_batches) // 2

        train_idx = [int(x) for x in range(train_batches * self.batch_size)]
        val_idx = [
            int(x)
            for x in range(
                (train_batches * self.batch_size),
                (train_batches * self.batch_size + val_batches * self.batch_size),
            )
        ]
        test_idx = [
            int(x)
            for x in range(
                (train_batches * self.batch_size + val_batches * self.batch_size),
                (train_batches * self.batch_size + val_batches * self.batch_size * 2),
            )
        ]
        return train_idx, val_idx, test_idx

    def _save_file(self, out_file: str, entries: list[str]) -> None:
        with open(out_file, "w") as f:
            for entry in entries:
                f.write(f"{entry}\n")

    def _save_dataset(self, split: str):
        out_file = f"{self.output_dir}/{split}_imgs.txt"
        # lbl_outfile = f"{self.output_dir}/{split}_lbls.txt"

        if split == "train":
            idx = self._train_idx
        elif split == "val":
            idx = self._val_idx
        elif split == "test":
            idx = self._test_idx

        else:
            logging.warning(
                f"{self.__class__.__name__}: invalid split provided, skipping saving dataset"
            )
            return

        imgs = [self._ds_images[x] for x in idx]
        self._save_file(out_file, imgs)

        """
        Labels are not saved atm, because they are read in before the split. Use image names to get the corresponding labels
        """
        # lbls = [self._ds_labels[x] for x in idx]
        # self._save_file(lbl_outfile, lbls) #

    def build_charset(self):
        """
        Returns a charset based on all provided labels. This can take a long time when a large amount
        of samples is used. Optimizing this function or using a pre-built char index might be an option depending on the scenario.
        """
        self._charset = build_charset(self._ds_labels)
        logging.info(
            f"No charset was provided, built one with {len(self._charset)} characters."
        )

    def get_train_data(self) -> tuple[list[str], list[str]]:
        images = [self._ds_images[x] for x in self._train_idx]
        labels = [self._ds_labels[x] for x in self._train_idx]
        return images, labels

    def get_val_data(self) -> tuple[list[str], list[str]]:
        images = [self._ds_images[x] for x in self._val_idx]
        labels = [self._ds_labels[x] for x in self._val_idx]
        return images, labels

    def get_test_data(self) -> tuple[list[str], list[str]]:
        images = [self._ds_images[x] for x in self._test_idx]
        labels = [self._ds_labels[x] for x in self._test_idx]

        return images, labels

    def get_charset(self) -> list[str]:
        if self._charset is None:
            logging.info(f"Charset is None, building charset..")
            self.build_charset()

        return self._charset


class OCRDataLoader(Sequence):
    def __init__(
        self,
        images: list[str],
        labels: list[str],
        charset: list[str],
        batch_size: int = 32,
        keep_channel_dim: bool = False,  # default for easter
        permute_hw: bool = True,  # default for easter
        pad_token: int = 0,
        max_output_length: int = 500,
    ):
        self._images = images
        self._labels = labels
        self._charset = charset
        self.img_width: int = 2000
        self.img_height: int = 80
        self.keep_channel_dim: bool = keep_channel_dim
        self._batch_size = batch_size
        self._max_output_length = max_output_length
        self._pad_token = pad_token
        self._image_reader = ImageReader(
            img_width=self.img_width,
            img_height=self.img_height,
            keep_channel=self.keep_channel_dim,
            permute_hw=permute_hw,
        )
        # self._vectorizer = TextVectorizer(charset=self._charset) # not used atm
        self._char_to_num = StringLookup(vocabulary=self._charset, mask_token=None)

    def get_charset(self):
        return self._charset

    def _read_img(self, img):
        tf_img = tf.io.read_file(img)
        tf_img = tf.image.decode_jpeg(tf_img, channels=1)
        tf_img = tf.image.resize_with_pad(tf_img, self.img_height, self.img_width)
        # tf_img = tf.where(tf_img > 180, 255, 0)
        tf_img = tf.cast(tf_img, tf.float32) / 255.0
        tf_img = tf.transpose(tf_img, perm=[1, 0, 2])

        if not self.keep_channel_dim:
            tf_img = tf.squeeze(tf_img)

        return tf_img

    def _vectorize_label(self, label):
        padding_token = 0
        vec_label = self._char_to_num(
            tf.strings.unicode_split(label, input_encoding="UTF-8")
        )
        length = tf.shape(vec_label)[0]
        pad_amount = (
            self._max_output_length - length
        )  # TODO: make sure this is not negtaive, maybe truncate label?
        vec_label = tf.pad(
            vec_label, paddings=[[0, pad_amount]], constant_values=padding_token
        )
        return vec_label

    def __len__(self):
        return int(np.ceil(len(self._images) / float(self._batch_size)))

    def __getitem__(self, idx):
        image_batch = self._images[
            idx * self._batch_size : (idx + 1) * self._batch_size
        ]
        label_batch = self._labels[
            idx * self._batch_size : (idx + 1) * self._batch_size
        ]

        gtTexts = np.ones([self._batch_size, self._max_output_length])
        input_length = np.ones((self._batch_size, 1)) * self._max_output_length
        label_length = np.zeros((self._batch_size, 1))

        # TODO: add some sensible handling of different architectures here, maybe use sublcassing to override this and have different classes for different architectures
        if not self.keep_channel_dim:
            imgs = np.ones([self._batch_size, self.img_width, self.img_height])
        else:
            imgs = np.ones(
                [self._batch_size, self.img_height, self.img_width, 1]
            )  # maybe use channe here?

        for idx in range(0, len(image_batch)):
            imgs[idx] = self._image_reader(image_batch[idx])
            # imgs[idx] = self._read_img(image_batch[idx])
            # lbl = self._label_reader(label_batch[idx])
            gtTexts[idx] = self._vectorize_label(label_batch[idx])
            label_length[idx] = len(label_batch[idx])
            input_length[idx] = self._max_output_length

        inputs = {
            "images": imgs,
            "labels": gtTexts,
            "input_length": input_length,
            "label_length": label_length,
        }

        outputs = {"ctc": np.zeros([self._batch_size])}

        return inputs, outputs


class OCRTrainer:
    """
    A wrapper class that handles the training based on the specified architecture
    # TODO:
    1. add a custom Callback for running a prediction from a sample from the test set
    2. add several parameters to allow more customization
    """

    def __init__(
        self,
        dataset: OCRDataset,
        architecture: str = "easter2",
        model_name: str = "model",
    ) -> None:
        self.architecture = architecture
        self.dataset = dataset
        self.train_images, self.train_labels = self.dataset.get_train_data()
        self.valid_images, self.valid_labels = self.dataset.get_val_data()
        # self.test_dataset = self.dataset.get_test_data()

        self.train_loader = OCRDataLoader(
            self.train_images,
            self.train_labels,
            batch_size=self.dataset.batch_size,
            charset=self.dataset.get_charset(),
        )
        self.val_loader = OCRDataLoader(
            self.valid_images,
            self.valid_labels,
            batch_size=self.dataset.batch_size,
            charset=self.dataset.get_charset(),
        )

        self.model_savepath = (
            f"{self.dataset.output_dir}/{model_name}_{architecture}.hdf5"
        )

        self.callbacks = [
            ModelCheckpoint(
                filepath=self.model_savepath,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min",
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=3, min_lr=1e-8, verbose=1
            ),
            EarlyStopping(monitor="val_loss", patience=5),
            LearningRateScheduler(self.lr_schedule),
        ]

        tf.keras.backend.clear_session()
        self.model = Easter2(classes=len(self.dataset.get_charset()))

        self._save_model_config()

    def lr_schedule(self, epoch, lr, epoch_limit: int = 30):
        """
        :param epoch: current training epoch
        :param lr: current learning rate
        :param epoch_limit: threshold above which the learning rate gets reduced, requires some experimentation
        :return: reduced learning rate
        """
        if epoch < epoch_limit:
            return lr
        else:
            new_lr = lr * tf.math.exp(-0.1)
            logging.info(f"scheduled  learning rate reduction: {new_lr}")
            return new_lr

    def _save_model_config(self):
        _charset = self.dataset.get_charset()

        if "[UNK]" in _charset:
            del _charset[_charset.index("[UNK]")]

        if "[BLK]" in _charset:
            del _charset[_charset.index("[BLK]")]

        _charset = "".join(x for x in _charset)

        network_config = {
            "model": self.model_savepath,
            "architecture": self.architecture,
            "input_width": self.train_loader.img_width,
            "input_height": self.train_loader.img_height,
            "charset": _charset,
        }

        out_file = os.path.join(self.dataset.output_dir, "model_config.json")
        json_out = json.dumps(network_config, ensure_ascii=False, indent=2)

        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(json_out)

        logging.info(f"Saved model config to: {out_file}")

    def _save_training_history(self, history):
        out_file = f"{self.dataset.output_dir}/train_history.txt"

        with open(out_file, "w") as f:
            f.write(str(history))

    def _load_weights(self, weights_file: str):
        try:
            self.model.load_weights(weights_file)
        except BaseException as e:
            logging.error(f"Failed to load weights: {e}")

    def train(self, epochs: int = 30):
        history = self.model.fit(
            self.train_loader,
            epochs=epochs,
            validation_data=self.val_loader,
            shuffle=True,
            callbacks=self.callbacks,
        )

        self._save_training_history(history.history)

        logging.info(f"Finished training of {epochs} epochs!")

    def fine_tune(
        self,
        weights_file: str,
        keep_layers: int = 7,
        epochs: int = 10,
        freeze_layers: bool = True,
    ) -> None:
        """
        Use this function for tine tuning a pre-trained network. Note that the character set has to match the one
        used for pretraining in number and sequence.

        weights_file: a .hdf5 weights file
        keep_layers: The number of layers from the top/end of the network that remain unfreezed. This value depends by and large on the architecture.
        freeze_layers: whether or not to freeze the specified layer range
        """
        logging.info(f"Loading weights: {weights_file}")
        self._load_weights(weights_file)

        if freeze_layers:
            for layer in self.model.layers[:-keep_layers]:
                layer.trainable = False

        self.train(epochs=epochs)
