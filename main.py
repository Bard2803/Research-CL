import argparse
from config import Config

from train import Trainer
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Continual Learning with PyTorch')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    main_path = os.path.dirname(os.path.abspath(__file__))

    # Load configuration
    config_path = os.path.join(main_path, "config.yaml")
    config = Config(config_path)

    trainer = Trainer(config)

    # Start training
    trainer.train()

    # Optionally: Evaluate the model after training
    # ...

if __name__ == "__main__":
    main()
