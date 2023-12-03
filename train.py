from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleCNN, MlpVAE 
import torch



class Trainer():
    def __init__(self, data_loader, config):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            torch.backends.cudnn.benchmark = True
            device_count = torch.cuda.device_count()
            print(f"Found {device_count} CUDA GPU devices.")
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.config = config
        self.data_loader = data_loader
        self.criterion = CrossEntropyLoss()
        self.model = SimpleCNN(num_classes=50).to(self.device)
        lr = self.config
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        #Line 1453 in thesis_code.py