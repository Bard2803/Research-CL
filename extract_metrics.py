from config import Config
import wandb
import os
import matplotlib.pyplot as plt
import pandas as pd

class Metrics():
    def __init__(self, config, group_name):
        wandb.login()
        api = wandb.Api()
        self.config = config
        project_name = config.get("wandb").get("project_name")
        self.runs = api.runs(project_name, filters = {"group": group_name})
        folder_name = config.get("wandb_metrics_extraction").get("folder_name")
        self.metrics_path = self.create_folder(folder_name)

    def create_folder(self, folder_name):
        # Check if the folder exists
        main_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(main_path, folder_name)
        if not os.path.exists(folder_path):
            # Create the folder if it does not exist
            os.makedirs(folder_path)
        return folder_path
    
    def num_epochs_per_experience(self):
        df = pd.DataFrame()
        counts = []
        counter = 0
        strategy_names = []
        # Fetch the logged metrics for each run
        for num_runs, run in enumerate(self.runs, 1):
            history = run.history()
            history = pd.DataFrame(history)
            counts = [] 
            strategy_names.append(run.name)
            for index, row in history.iterrows():
                if isinstance(row['TrainingExperience'], float) and row['TrainingExperience'] > 0:
                    counter += 1
                    # Append the count to the list and reset the counter
                    counts.append(counter)
                    counter = 0
                else:
                    counter += 1
            df[run.name + str(strategy_names.count(run.name))] = counts
        print("COUNTS", df.head(10))
        history.to_excel(os.path.join(self.metrics_path, "convergence_output.xlsx"))

        return num_runs, df
    
    def plot_bar_chart(self, summary, print_plot=False):
        # Plot
        plt.figure(figsize=(10, 5))
        bars = plt.bar(summary['strategy'], summary['mean'], yerr=summary['std'], capsize=7, color='skyblue', edgecolor='black')

        # Adding titles and labels
        plt.title('Mean and Standard Deviation of number of epochs to convergence')
        plt.xlabel('Strategy')
        plt.ylabel('Number of epochs')

        # Adding numerical values at the top of each bar
        for bar, value in zip(bars, summary['mean']):
            plt.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 3, value),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # Show plot
        plt.savefig(os.path.join(self.metrics_path,"strategies.png"))

        # Print the summary DataFrame
        if print_plot:
            print(summary)
        
    
    def calculate_convergence(self, num_runs, df):
        strategies = ["Cumulative", "GR", "EWC", "GEM", "CWR*", "Naive"]
        # Calculate the mean and standard deviation for each strategy
        columns = ['strategy', 'mean', 'std']
        summary = pd.DataFrame(columns=columns)
        for strategy in strategies:
            # For mean and standard deviation of sum of epochs
            columns = [col for col in df.columns if col.startswith(strategy)]
            strategy_data = df[columns]
            values = strategy_data.values
            mean = values.mean()
            std = values.std()
            summary = pd.concat([summary, pd.DataFrame([{"strategy": strategy, "mean": mean, "std": std}])], ignore_index=True)

        #     # For number of epochs
        #     strategy_data = []
        #     for i in range(num_runs):
        #         Cumulative = f'{strategy}{i}'
        #         if Cumulative in df:
        #             strategy_data.extend(df[Cumulative])
        #     plt.plot(range(len(strategy_data)), strategy_data, marker='o', label=strategy)

        # # For number of epochs
        # # plt.axhline(y=PATIENCE, color='red', linewidth=2, linestyle='--', label='Patience')
        # plt.xlabel('Experience')
        # plt.ylabel('Number of epochs')
        # plt.title('Number of epochs in each experience for class incremental scenario')
        # # Set custom tick labels
        # total_iterations = num_runs * 10  # 10 runs, 10 counts for each experience
        # tick_positions = list(range(total_iterations))
        # tick_labels = [str(i % 10) for i in range(total_iterations)]
        # plt.xticks(tick_positions, tick_labels)
        # plt.xticks(range(len(strategy_data)))
        # plt.legend()

        # # Show plot
        # plt.tight_layout()
        # plt.grid(True)
        # plt.savefig(os.path.join(self.metrics_path, "epochs.png"))
        self.plot_bar_chart(summary)
        wandb.finish()
        print("finished extracting convergence")


    def extract_convergence(self):
        num_runs, df = self.num_epochs_per_experience()
        self.calculate_convergence(num_runs, df)



if __name__ == "__main__":
    main_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(main_path, "config.yaml")
    config = Config(config_path)
    group_name = "splitmnist_2023-12-21 22:12:29.655607"
    metrics = Metrics(config, group_name)
    metrics.extract_convergence()

    # Call method for appropriate metrics extraction