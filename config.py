import yaml

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.data = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        return self.data.get(key, default)

# Usage
config = Config('config.yaml')
learning_rate = config.get('training').get('learning_rate')
print(f"Learning Rate: {learning_rate}")
