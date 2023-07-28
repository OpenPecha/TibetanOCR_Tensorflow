import os
import re
import random
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm


def get_basename(x):
    return os.path.basename(x).split(".")[0]


def shuffle_data(images: list, labels: list) -> tuple[list, list]:
    c = list(zip(images, labels))
    random.shuffle(c)
    a, b = zip(*c)
    return list(a), list(b)


def validate_data(images_paths: list[str], label_paths: list[str]):
    """
    Attempts to remove small and empty files, builds a common list if images and labels if there is a mismatch between both
    """
    # filters empty text files and white pages
    images = [x for x in images_paths if os.stat(x).st_size >= 3000]
    labels = [x for x in label_paths if os.stat(x).st_size != 0]

    image_list = list(map(get_basename, images))
    transcriptions_list = list(map(get_basename, labels))

    return list(set(image_list) & set(transcriptions_list))


def clean_unicode_label(l, full_bracket_removal: bool = False):
    """
    Some preliminary clean-up rules for the Unicode text.
    - Note: () are just removed. This was valid in case of the Lhasa Kanjur.
    In other e-texts, a complete removal of the round and/or square brackets together with the enclosed text should be applied
    in order to remove interpolations, remarks or similar additions.
    In such cases set full_bracket_removal to True.
    """

    l = re.sub("[\uf8f0]", " ", l)
    l = re.sub("[༌]", "་", l)  # replace triangle tsheg with regular

    if full_bracket_removal:
        l = re.sub("[\[(].*?[\])]", "", l)
    else:
        l = re.sub("[()]", "", l)
    return l


def post_process_wylie(l):
    l = l.replace("\\u0f85", "&")
    l = l.replace("\\u0f09", "ä")
    l = l.replace("\\u0f13", "ö")
    l = l.replace("\\u0f12", "ü")
    l = l.replace("_", " ")
    l = l.replace("  ", " ")

    return l


def read_data2(
    image_list,
    label_list: list,
    converter,
    min_label_length: int = 30,
    max_label_length: int = 240,
) -> tuple[list[str], list[str]]:
    """
    Reads all labels into memory, filter labels for min_label_length and max_label_length.
    # TODO:
    1) convert the training labels to wylie ahead of training and clean them up avoiding multiple checks while reading the dataset
    2) read and encode them more efficiently via tf.keras.layers.TextVectorization

    """
    labels = []
    images = []
    for image_path, label_path in tqdm(
        zip(image_list, label_list), total=len(label_list), desc="reading labels"
    ):

        f = open(label_path, "r", encoding="utf-8")
        label = f.readline()
        label = clean_unicode_label(label)

        if min_label_length < len(label) < max_label_length:
            label = converter.toWylie(label)
            label = post_process_wylie(label)

            # filter everything that was not converted by pyewts or replaced by post-processing
            if "\\u" not in label:
                labels.append(label)
                images.append(image_path)

    return images, labels


def get_split_label(x):
    return set(x)


def build_charset(labels: list):
    """
    Builds a charset from list of unicode labels.
    :param labels: list of unicode labels
    :return: character set to be used for training
    """
    split_labels = [get_split_label(x) for x in labels]
    flattened_labels = [x for xs in split_labels for x in xs]
    charset = set(flattened_labels)

    charset = sorted(charset)
    charset.append("[BLK]")
    charset.insert(0, "[UNK]")

    return charset


def get_charset(characters: str):
    """
    To be used when reading a charset from a config file
    """
    characters = [c for c in characters]
    characters.append("[BLK]")
    characters.insert(0, "[UNK]")

    return characters


class ImageReader:
    def __init__(
        self,
        img_width: int = 2000,
        img_height: int = 80,
        keep_channel: bool = True,
        permute_hw: bool = True,
    ) -> None:
        self.img_width = img_width
        self.img_height = img_height
        self.keep_channel = keep_channel
        self.permute_hw = permute_hw

    def __call__(self, img):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize_with_pad(img, self.img_height, self.img_width)
        img = tf.cast(img, tf.float32) / 255.0

        if self.permute_hw:
            img = tf.transpose(img, perm=[1, 0, 2])

        if not self.keep_channel:
            img = tf.squeeze(img)

        return img


def save_train_data(
    data: list, out_path: str, split: str = "train", data_type: str = "images"
) -> None:
    out_file = os.path.join(f"{out_path}/{split}-{type}.txt")

    with open(out_file, "w", encoding="UTF-8") as f:
        for x in data:
            x_name = os.path.basename(x).split(".")[0]
            f.write(f"{x_name}\n")


class TextVectorizer:
    def __init__(
        self,
        charset: list[str],
        max_sequence_length: int = 500,
        padding_token: int = 0,
        pad_sequences: bool = True,
    ) -> None:
        self.charset = charset
        self.sequence_length = max_sequence_length
        self.padding_token = padding_token
        self.pad_sequences = pad_sequences

    def __call__(self, item):
        vec_label = [x for x in item]
        vec_label = [self.charset.index(x) for x in vec_label]

        if self.pad_sequences:
            length = tf.shape(vec_label)[0]
            pad_amount = self.sequence_length - length
            vec_label = tf.pad(
                vec_label,
                paddings=[[0, pad_amount]],
                constant_values=self.padding_token,
            )
        return vec_label


def decode_label(label: np.array, charset: list[str], converter) -> str:
    label = label.astype(np.uint8)
    label = np.delete(label, np.where(label == 0))
    label = "".join(charset[x] for x in label)
    label = label.replace("§", " ")
    label = converter.toUnicode(label)

    return label


def decode_image(image: np.array) -> np.array:
    image = np.transpose(image, axes=(1, 0))
    image *= 255
    image = image.astype(np.uint8)

    return image
