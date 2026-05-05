import argparse

from ae_train import train_model
from ae_test import test_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="Choose whether to train, test, or run both.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        test_model()
    else:
        train_model()
        test_model()


if __name__ == "__main__":
    main()