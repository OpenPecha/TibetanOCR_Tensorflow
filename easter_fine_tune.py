import sys
import json
import os.path
import argparse
from src.Utils import get_charset
from src.Modules import OCRDataset, OCRTrainer

"""
run the pipeline with e.g.: python easter_fine_tune.py --dataset_dir "Data/KhyentseWangpo" --config "Models/LhasaKanjur/model_config.json" --epochs 10
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--epochs", type=int, required=False, default=30)
    parser.add_argument("--model_name", type=str, required=False, default="fine_tuned_model")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    if not os.path.isdir(dataset_dir):
        sys.exit(f"'{dataset_dir}' is not a valid directory, cancelling training.")

    batch_size = args.batch_size
    epochs = args.epochs

    f = open(args.config, encoding="utf-8")
    model_config = json.load(f)

    charset = get_charset(model_config["charset"])

    model_name = args.model_name

    ocr_dataset = OCRDataset(dataset_dir, batch_size=batch_size, charset=charset)
    ocr_trainer = OCRTrainer(
        ocr_dataset, architecture=model_config["architecture"], model_name=model_name
    )
    ocr_trainer.fine_tune(model_config["model"], epochs=args.epochs)
