import argparse
from config import Config
from data_loader import DataLoader
from train import Trainer
import os
import logging

def setup_logging():
    logging.basicConfig(filename='logs.log', level=logging.DEBUG, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Continual Learning with PyTorch')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    setup_logging()
    logging.info("Application started")

    main_path = os.path.dirname(os.path.abspath(__file__))

    # Load configuration
    config_path = os.path.join(main_path, "config.yaml")
    config = Config(config_path)


    # Data loading
    data_loader = DataLoader(config)

    # Training setup
    trainer = Trainer(data_loader, config)

    # Start training
    trainer.train()

    # Optionally: Evaluate the model after training
    # ...

if __name__ == "__main__":
    main()
