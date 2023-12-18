from config import Config
import wandb
import os
import matplotlib.pyplot as plt
import pandas as pd

class Metrics():
    def __init__(self, config):
        wandb.login()
        api = wandb.Api()
        self.config = config
        project_name = config.get("wandb").get("project_name")
        self.runs = api.runs(project_name)
        self.metrics_path = self.create_folder("metrics_extraction")

    def create_folder(self, folder_name):
        # Check if the folder exists
        main_path = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(main_path, folder_name)
        if not os.path.exists(folder_path):
            # Create the folder if it does not exist
            os.makedirs(folder_path)
        return folder_path

    def extract_convergence(self):
        # Initialize empty DataFrame
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
                if str(row['TrainingExperience']).split('.')[0].isdigit():
                    counter += 1
                    # Append the count to the list and reset the counter
                    counts.append(counter)
                    counter = 0
                else:
                    counter += 1
            # print(len(counts))
            df[run.name + str(strategy_names.count(run.name))] = counts

        print("COUNTS", df.head(10))

        history.to_excel(os.path.join(self.metrics_path, "convergence_output.xlsx"))

        # Strategies names without version numbers
        strategies = ['Cumulative', 'EWC', 'GEM', 'CWR', 'Naive']

        # Calculate the mean and standard deviation for each strategy
        summary = {}
        fig, ax = plt.subplots(figsize=(12, 6))
        for strategy in strategies:
            # For mean and standard deviation of sum of epochs
            columns = [col for col in df.columns if col.startswith(strategy)]
            strategy_data = df[columns]
            values = strategy_data.values
            mean = values.mean()
            std = values.std()
            summary[strategy] = {'mean': mean, 'std': std}


            # For number of epochs
            strategy_data = []
            for i in range(num_runs):
                Cumulative = f'{strategy}{i}'
                if Cumulative in df:
                    strategy_data.extend(df[Cumulative])
            plt.plot(range(len(strategy_data)), strategy_data, marker='o', label=strategy)

        # For number of epochs
        # plt.axhline(y=PATIENCE, color='red', linewidth=2, linestyle='--', label='Patience')
        plt.xlabel('Experience')
        plt.ylabel('Number of epochs')
        plt.title('Number of epochs in each experience for class incremental scenario')
        # Set custom tick labels
        total_iterations = num_runs * 10  # 10 runs, 10 counts for each experience
        tick_positions = list(range(total_iterations))
        tick_labels = [str(i % 10) for i in range(total_iterations)]
        plt.xticks(tick_positions, tick_labels)
        plt.xticks(range(len(strategy_data)))
        plt.legend()

        # Show plot
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(self.metrics_path, "epochs.png"))


        # For mean and standard deviation of sum of epochs
        # Convert the summary to a DataFrame for better visualization
        summary_df = pd.DataFrame(summary).T

        # Plot
        plt.figure(figsize=(10, 5))
        bars = plt.bar(summary_df.index, summary_df['mean'], yerr=summary_df['std'], capsize=7, color='skyblue', edgecolor='black')

        # Adding titles and labels
        plt.title('Mean and Standard Deviation of number of epochs to convergence')
        plt.xlabel('Strategy')
        plt.ylabel('Number of epochs')

        # Adding numerical values at the top of each bar
        for bar, value in zip(bars, summary_df['mean']):
            plt.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 3, value),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # Show plot
        plt.savefig(os.path.join(self.metrics_path,"strategies.png"))

        # Print the summary DataFrame
        print(summary_df)
        wandb.finish()

        print("finished extracting convergence")


if __name__ == "__main__":
    main_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(main_path, "config.yaml")
    config = Config(config_path)
    metrics = Metrics(config)
    metrics.extract_convergence()

    # Call method for appropriate metrics extraction