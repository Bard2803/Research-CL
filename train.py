from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleCNN, MlpVAE 
import torch
from avalanche.training.supervised import Naive, Cumulative, EWC, GenerativeReplay, VAETraining, Replay, GEM, PNNStrategy, CWRStar
from evaluation import Evaluation
from avalanche.training.plugins import EarlyStoppingPlugin



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
        print(f">>>>> Using {self.device} for training <<<<<<")

        self.data_loader = data_loader
        self.config = config
        self.num_classes = config.get("model").get("num_classes")
        self.init_model()

    def init_model(self):
        self.model = SimpleCNN(num_classes=self.num_classes).to(self.device)
        lr = self.config.get("training").get("learning_rate")
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = CrossEntropyLoss()
        #Line 1453 in thesis_code.py

    def train(self):
        num_runs = self.config.get("training").get("num_runs")
        batchsize_train = self.config.get("training").get("batch_size")
        batchsize_eval = self.config.get("training").get("batch_size")
        epochs = self.config.get("training").get("epochs")
        fraction_to_take = self.config.get("dataset").get("fraction_to_take")
        Evaluator = Evaluation(self.config.load_config())
        eval_every = self.config.get("training").get("eval_every")
        patience = self.config.get("training").get("patience")
        train_stream = self.data_loader.get_train_stream()
        val_stream = self.data_loader.get_val_stream()
        test_stream = self.data_loader.get_test_stream()
        strategies = {"Naive": Naive, "CWRStar": CWRStar, "GEM": GEM, "EWC": EWC, "Cumulative": Cumulative}
        for j in range(num_runs):
            for strategy_name, strategy in strategies.items():
                # reinitilizate the model each run to reset its parameters and avoid carry over effect
                self.init_model()
                Evaluator.create_evaluator(strategy_name)
                eval_plugin = Evaluator.get_eval_plugin()

                cl_strategy = strategy(
                    self.model, self.optimizer, self.criterion, device=self.device,
                    train_mb_size=batchsize_train, train_epochs=epochs, eval_mb_size=batchsize_eval, evaluator=eval_plugin, \
                    eval_every=eval_every, plugins=[EarlyStoppingPlugin(patience, "valid")])
                

                print(f"Current training strategy: {cl_strategy}")
                for train_experience, val_experience in zip(train_stream, val_stream):
                    print(f"Experience number train {train_experience.current_experience}")
                    print(f"classes in this experience train {train_experience.classes_in_this_experience}")
                    print(f"Classes seen so far train {train_experience.classes_seen_so_far}")
                    print(f"Number of examples before fraction take {len(train_experience.dataset)}")
                    train_experience.dataset = train_experience.dataset.subset(range(int(len(train_experience.dataset)*fraction_to_take)))
                    print(f"Training on {len(train_experience.dataset)} examples")

                    cl_strategy.train(train_experience, eval_streams=[val_experience])

                    cl_strategy.eval(test_stream)