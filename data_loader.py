from avalanche.benchmarks.generators import benchmark_with_validation_stream, class_balanced_split_strategy, random_validation_split_strategy, nc_benchmark
from avalanche.benchmarks.classic import CORe50
from avalanche.benchmarks.datasets import CORe50Dataset
from torchvision import transforms
import logging 

def transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.dataset = None
        self.prepare_dataset()

    def prepare_core50(self):
        val_fraction = self.config.get("dataset").get("validation_fraction")
        scenario = self.config.get("scenario").get("type")
        if scenario == "ni":
            core50_train = CORe50(train=True, scenario="ni", mini=True, object_lvl=True)
            core50_test = CORe50(train=False, scenario="ni", mini=True, object_lvl=True)
            f = lambda exp: random_validation_split_strategy(val_fraction, exp)
        elif scenario == "nc":
            # Load the CORe50 dataset
            core50_train = CORe50Dataset(train=True, mini=True)
            core50_test = CORe50Dataset(train=False, mini=True)
            # Create different split
            core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())
            f = lambda exp: class_balanced_split_strategy(val_fraction, exp)
        else:
            raise NameError("Scenario name unknown")
        return benchmark_with_validation_stream(core50, custom_split_strategy=f)

    def prepare_dataset(self):
        dataset_name = self.config.get("dataset").get("name")
        if dataset_name == "core50":
            self.dataset = self.prepare_core50()
        else:
            # Handle other datasets or raise an error
            raise ValueError(f"Dataset '{dataset_name}' not supported")
        
        if self.dataset:
            self.logger.info(f'--- Task labels: {self.dataset.task_labels}')
        else:
            self.logger.warning('Dataset is not properly initialized')

    def get_train_stream(self):
        if self.dataset:
            return self.dataset.train_stream
        raise RuntimeError('Dataset is not initialized')

    def get_val_stream(self):
        if self.dataset:
            return self.dataset.valid_stream
        raise RuntimeError('Dataset is not initialized')

    def get_test_stream(self):
        if self.dataset:
            return self.dataset.test_stream
        raise RuntimeError('Dataset is not initialized')