from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleCNN, MlpVAE
import torch
from avalanche.training.supervised import Naive, Cumulative, EWC, GenerativeReplay, VAETraining, Replay, GEM, PNNStrategy, CWRStar
from evaluation import Evaluation
from avalanche.training.plugins import EarlyStoppingPlugin, GenerativeReplayPlugin
from data_loader import DataLoader
from models import SimpleCNNGrayScale
from utils import *



class Trainer():
    def __init__(self, config):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            torch.backends.cudnn.benchmark = True
            device_count = torch.cuda.device_count()
            print(f"Found {device_count} CUDA GPU devices.")
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f">>>>> Using {self.device} for training <<<<<<")
        self.config = config
        self.num_classes = config.get("model").get("num_classes")

    def init_model(self, dataset):
        if dataset == "splitmnist":
            self.model = SimpleCNNGrayScale(num_classes=self.num_classes).to(self.device)
        else:
            self.model = SimpleCNN(num_classes=self.num_classes).to(self.device)
        self.lr = self.config.get("training").get("learning_rate")
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = CrossEntropyLoss()
        #Line 1453 in thesis_code.py

    def Generator_Strategy(self, n_classes, data_sample_shape, lr, batchsize_train, batchsize_eval, epochs, nhid, ES_plugin, dataset):
        
        if dataset == "splitmnist":
            # First argument is shape of input sample so 1 for GrayScale 32x32 for image resolution
            generator = MlpVAE(data_sample_shape, nhid, n_classes, device=self.device)
        else:
            # First argument is shape of input sample so 3 for RGB 32x32 for image resolution
            generator = MlpVAE(data_sample_shape, nhid, n_classes, device=self.device)
        optimizer = Adam(generator.parameters(), lr=lr)
        # optimzer:

        # strategy (with plugin):
        generator_strategy = VAETraining(
            model=generator,
            optimizer=optimizer,
            train_mb_size=batchsize_train,
            train_epochs=epochs,
            eval_mb_size=batchsize_eval,
            device=self.device,
            plugins=[
                GenerativeReplayPlugin(),
                ES_plugin
            ],
        )

        return generator_strategy
    
    def get_dataset(self, dataset_name, scenario):
        data_loader = DataLoader(self.config, dataset_name, scenario)
        train_stream = data_loader.get_train_stream()
        val_stream = data_loader.get_val_stream()
        test_stream = data_loader.get_test_stream()

        return train_stream, val_stream, test_stream
    
    def generate_benchmarks_list(self):
        datasets = self.config.get("dataset").get("name")
        scenarios = self.config.get("scenario").get("type")
        benchmarks_list = []
        for dataset in datasets:
            if dataset == "core50":
                for scenario in scenarios:
                    benchmarks_list.append((dataset, scenario))
            else:
                benchmarks_list.append((dataset, None))
        return benchmarks_list

    def train(self):
        num_runs = self.config.get("training").get("num_runs")
        batchsize_train = self.config.get("training").get("batch_size")
        batchsize_eval = self.config.get("training").get("batch_size")
        epochs = self.config.get("training").get("epochs")
        fraction_to_take = self.config.get("dataset").get("fraction_to_take")
        eval_every = self.config.get("training").get("eval_every")
        patience = self.config.get("training").get("patience")
        strategies = {"Naive": Naive, "CWR*": CWRStar, "GEM": GEM, "EWC": EWC, "GR": GenerativeReplay, "Cumulative": Cumulative}
        benchmarks = self.generate_benchmarks_list()
        for dataset, scenario in benchmarks:
            train_stream, val_stream, test_stream = self.get_dataset(dataset, scenario)
            data_sample_shape = train_stream[0].dataset[0][0].shape
            Evaluator = Evaluation(self.config, dataset, scenario)
            for j in range(num_runs):
                for strategy_name, strategy in strategies.items():
                    # reinitilizate the model each run to reset its parameters and avoid carry over effect
                    self.init_model(dataset)
                    Evaluator.create_evaluator(strategy_name)
                    eval_plugin = Evaluator.get_eval_plugin()

                    if strategy_name == "CWR*":
                        cl_strategy = strategy(
                        self.model, self.optimizer, self.criterion, cwr_layer_name='classifier.0', device=self.device,
                        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
                        eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid")])

                    elif strategy_name == "GEM":
                        cl_strategy = strategy(
                        self.model, self.optimizer, self.criterion, device=self.device, patterns_per_exp=1024, memory_strength=1,
                        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
                        eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid")])

                    elif strategy_name == "EWC":
                        cl_strategy = strategy(
                        self.model, self.optimizer, self.criterion, ewc_lambda=100, device=self.device,
                        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
                        eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid")])

                    elif strategy_name == "GR":
                        generator_strategy = self.Generator_Strategy(self.num_classes, data_sample_shape, self.lr, batchsize_train, batchsize_eval,
                        epochs, 2, EarlyStoppingPlugin(patience, "valid"), dataset)
                        
                        cl_strategy = strategy(
                        self.model, self.optimizer, self.criterion, device=self.device,
                        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin,
                        eval_every=eval_every, generator_strategy=generator_strategy, plugins=[EarlyStoppingPlugin(patience, "valid")])

                    else:
                        cl_strategy = strategy(
                        self.model, self.optimizer, criterion=self.criterion, device=self.device, 
                        train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval,
                        evaluator=eval_plugin, eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid")])

                    print(f"Current training strategy: {cl_strategy}")
                    for train_experience, val_experience in zip(train_stream, val_stream):
                        print(f"Experience number train {train_experience.current_experience}")
                        print(f"classes in this experience train {train_experience.classes_in_this_experience}")
                        print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
                        print(f"Number of train examples before fraction take {len(train_experience.dataset)}")
                        print(f"Number of val examples before fraction take {len(val_experience.dataset)}")
                        # train_experience.dataset = train_experience.dataset.subset(range(int(len(train_experience.dataset)*fraction_to_take)))
                        # val_experience.dataset = val_experience.dataset.subset(range(int(len(val_experience.dataset)*fraction_to_take)))
                        train_experience.dataset = load_subset_train_or_val(train_experience, fraction_to_take)
                        val_experience.dataset = load_subset_train_or_val(val_experience, fraction_to_take)
                        print(f"Training on {len(train_experience.dataset)} examples")
                        print(f"Validating on {len(val_experience.dataset)} examples")

                        cl_strategy.train(train_experience, eval_streams=[val_experience])

                        cl_strategy.eval(test_stream)