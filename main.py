import argparse
from config import Config
from train import Trainer
import os

def main():
    parser = argparse.ArgumentParser(description='Continual Learning with PyTorch')
    parser.add_argument('--api_key', type=str, help='api key to wandb.')
    args = parser.parse_args()

    main_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(main_path, "config.yaml")
    config = Config(config_path)

    trainer = Trainer(config)

    trainer.train()

if __name__ == "__main__":
    main()
