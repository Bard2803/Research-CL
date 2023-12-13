from avalanche.benchmarks.generators import benchmark_with_validation_stream, class_balanced_split_strategy, random_validation_split_strategy, nc_benchmark
from avalanche.benchmarks.classic import CORe50, SplitMNIST, SplitCIFAR10
from torchvision import transforms

def transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

class DataLoader:
    def __init__(self, config, dataset_name, scenario):
        self.config = config 
        self.dataset = self.prepare_dataset(dataset_name, scenario)

    def prepare_core50(self, scenario, val_fraction):
        mini = self.config.get("scenario").get("mini")
        object_lvl = self.config.get("scenario").get("object_lvl")
        if scenario == "ni":
            core50 = CORe50(scenario=scenario, mini=mini, object_lvl=object_lvl)
            f = lambda exp: random_validation_split_strategy(val_fraction, False, exp)
        elif scenario == "nc":
            # Load the CORe50 dataset
            # core50_train = CORe50Dataset(train=True, mini=mini, object_level=object_lvl)
            # core50_test = CORe50Dataset(train=False, mini=mini, object_level=object_lvl)
            # # Create different split
            # core50 = nc_benchmark(core50_train, core50_test, n_experiences=10, shuffle=False, task_labels=False, train_transform=transform(), eval_transform=transform())
            core50 = CORe50(scenario=scenario, mini=mini, object_lvl=object_lvl)
            f = lambda exp: class_balanced_split_strategy(val_fraction, exp)
        else:
            raise NameError("Scenario name unknown")
        
        core50 = benchmark_with_validation_stream(core50, custom_split_strategy=f)
        return core50
    
    def prepare_splitmnist(self, num_experiences, f):
        splitMNIST = SplitMNIST(n_experiences=num_experiences, return_task_id=True)
        splitMNIST = benchmark_with_validation_stream(splitMNIST, custom_split_strategy=f)
        return splitMNIST
    
    def prepare_splitcifar10(self, num_experiences, f):
        splitCIFAR10 = SplitCIFAR10(n_experiences=num_experiences, return_task_id=True)
        splitCIFAR10 = benchmark_with_validation_stream(splitCIFAR10, custom_split_strategy=f)
        return splitCIFAR10

    def prepare_dataset(self, dataset_name, scenario):
        val_fraction = self.config.get("dataset").get("validation_fraction")
        num_experiences = self.config.get("scenario").get("num_experiences")
        f = lambda exp: class_balanced_split_strategy(val_fraction, exp)
        if dataset_name == "core50":
            return self.prepare_core50(scenario, val_fraction)
        if dataset_name == "splitmnist":
            return self.prepare_splitmnist(num_experiences, f)
        if dataset_name == "splitcifar10":
            return self.prepare_splitcifar10(num_experiences, f)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported")

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