import sys
import logging
import os.path
import argparse
from src.Modules import OCRDataset, OCRTrainer
from src.Utils import get_charset
from config import DEFAULT_CHARSET

"""
run the pipeline with e.g.:  python easter_training.py --dataset_dir "Data/DergeTenjur" --model_name "DergeTenjur" --optimizer "Lion"
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--epochs", type=int, required=False, default=30)
    parser.add_argument(
        "--architecture", choices=["easter2"], required=False, default="easter2"
    )
    parser.add_argument("--optimizer", choices=["Adam", "RMSProp", "Lion"])
    parser.add_argument("--model_name", type=str, required=False, default="ocr_model")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    epochs = args.epochs
    architecture = args.architecture
    optimizer = args.optimizer
    model_name = args.model_name

    charset = get_charset(DEFAULT_CHARSET)

    if not os.path.isdir(dataset_dir):
        sys.exit(f"'{dataset_dir}' is not a valid directory, cancelling training.")

    ocr_dataset = OCRDataset(dataset_dir, batch_size=batch_size, charset=charset)
    ocr_trainer = OCRTrainer(
        ocr_dataset, architecture=architecture, model_name=model_name, optimizer=optimizer
    )

    ### use this for training a network from scratch
    logging.info("starting training....")
    ocr_trainer.train(epochs=epochs)
