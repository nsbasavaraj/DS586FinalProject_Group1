import argparse
from train import train_model
from test import test_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Choose whether to train, test, or do both.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        test_model()
    elif args.mode == "both":
        train_model()
        test_model()


if __name__ == "__main__":
    main()