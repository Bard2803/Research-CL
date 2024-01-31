from config import Config
import wandb
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Metrics():
    def __init__(self, config, group_name):
        wandb.login()
        api = wandb.Api()
        self.config = config
        project_name = config.get("wandb").get("project_name")
        self.group_name = group_name
        # wandb.init(project=project_name, group=group_name)
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
        num_experiences = self.config.get("scenario").get("num_experiences")
        # Fetch the logged metrics for each run
        for num_runs, run in enumerate(self.runs, 1):
            history = run.history()
            history = pd.DataFrame(history)
            counts = [] 
            strategy_names.append(run.name)
            for index, row in history.iterrows():
                if not np.isnan(row['TrainingExperience']):
                    counter += 1
                    # Append the count to the list and reset the counter
                    counts.append(counter)
                    counter = 0
                else:
                    counter += 1
            while len(counts) != num_experiences:
                counts.append(sum(counts)/len(counts))
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

    
    def save_metrics_xlsx(self):
        data = pd.DataFrame()

        # Fetch the logged metrics for each run
        for run in self.runs:
            history = run.history(stream="system")
            history['run_id'] = run.id
            history['strategy_name'] = run.name
            data = pd.concat([data, history], ignore_index=True)

        data.to_excel(os.path.join(self.metrics_path, "system_metrics.xlsx"), index=False)
        data = data.interpolate()

        return data


    def extract_system_metrics(self, metric, description):
        data = self.save_metrics_xlsx()
        # Create a helper column to detect changes in strategy_name
        data['index_run'] = data.groupby("run_id").cumcount()

        df_mean = data.groupby(["strategy_name", "index_run"])[[metric, "_runtime"]].mean().reset_index()
        df_std = data.groupby(["strategy_name", "index_run"])[metric].std().reset_index()
        df_std['_step'] = data.groupby(["strategy_name", "index_run"])["_runtime"].mean().reset_index()["_runtime"]

        # Pivot data to have strategies as columns
        pivot_table = df_mean.pivot(index='_runtime', columns='strategy_name', values=metric)
        pivot_table['the_index'] = range(len(pivot_table))
        pivot_table_std = df_std.pivot(index='_step', columns='strategy_name', values=metric)

        # akima for gpu.0.gpu and cpu, linear for gpu.0.temp
        pivot_table.interpolate(method='linear', inplace=True, limit=20)
        pivot_table_std.interpolate(method='linear', inplace=True, limit=20)
        
        # Plotting
        pivot_table.drop(columns="the_index", inplace=True)

        plt.figure(figsize=(12, 6))

        # Define colors for each strategy
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # Loop through each strategy
        for i, strategy in enumerate(pivot_table.columns):
            means = pivot_table[strategy]
            stds = pivot_table_std[strategy] 
            x = pivot_table.index / (60*60)
            # Plot means for this strategy with customizations
            plt.plot(x, means,
                    label=strategy,
                    linewidth=2,
                    color=colors[i % len(colors)])
            
            # Add variance as shadowed region
            plt.fill_between(x, means - stds, means + stds,
                            color=colors[i % len(colors)], alpha=0.2)

        # Adding labels and title
        plt.xlabel('Runtime (h)')
        plt.ylabel(f'Mean {description}')
        plt.title(f'Mean {description} with Standard Deviation for Different Strategies')
        plt.legend()
        
        plot_name =  description + " " + self.group_name + ".png"
        path_to_plot = os.path.join(self.metrics_path, plot_name)
        plt.savefig(path_to_plot)
        # wandb.log({plot_name[:-4]: wandb.Image(path_to_plot)}, commit=True)

    def extract_system_metrics_all(self):
        all = {"system.gpu.0.powerWatts": "GPU Power Usage (W)", "system.gpu.0.gpu": "GPU Utilization",\
                          "system.gpu.0.temp": "GPU Temperature (°C)", "system.cpu": "CPU Utilization (%)",\
                            "system.memory": "System Memory Utilization (%)"}
        for metric, description in all.items():
            self.extract_system_metrics(metric, description)


    def extract_energy_consumption(self):
        data = self.save_metrics_xlsx()

        # Create a helper column to detect changes in strategy_name
        data['index_run'] = data.groupby("run_id").cumcount()

        df_mean= data.groupby(["strategy_name", "index_run"]).agg(
        mean_metric=("system.gpu.0.powerWatts", 'mean'),
        std_metric=("system.gpu.0.powerWatts", 'std'),
        mean_runtime=('_runtime', 'mean')
        ).reset_index()

        # Pivot data to have strategies as columns
        pivot_table = df_mean.pivot(index='mean_runtime', columns='strategy_name', values='mean_metric')
        pivot_table.interpolate(inplace=True, limit=30)

        pivot_table_std = df_mean.pivot(index='mean_runtime', columns='strategy_name', values='std_metric')
        pivot_table_std.interpolate(inplace=True, limit=30)

        # Lists to store area and standard deviation values
        energy_list = []
        std_devs = []

        # Loop through each strategy
        for strategy in pivot_table.columns:
            means = pivot_table[strategy].dropna()
            x = means.index

            stds = pivot_table_std[strategy].dropna()
            x_stds = stds.index
            
            # Calculate area under the curve using the trapezoidal rule
            energy = np.trapz(means, x) / 1e6
            std_dev = np.trapz(stds, x_stds)
            
            # Append area and standard deviation values to the lists
            energy_list.append(energy)
            std_devs.append(std_dev)

        # Create a DataFrame to hold the data for the bar plot
        bar_data = pd.DataFrame({
            'Strategy': pivot_table.columns,
            'Area': energy_list,
            'Std Dev': std_devs
        })

        # Sort bar_data in descending order based on 'Area'
        bar_data = bar_data.sort_values(by='Area', ascending=False)

        # Plotting
        plt.figure(figsize=(10, 6))

        # Bar plot for energy_list and standard deviations
        bars = plt.bar(bar_data['Strategy'], bar_data['Area'], capsize=5)

        # Bar plot for energy_list and standard deviations
        plt.bar(bar_data['Strategy'], bar_data['Area'], capsize=5)

        # Adding labels and title
        plt.xlabel('Strategy')
        plt.ylabel('Energy (MJ)')
        description = 'GPU energy used for training for different strategies'
        plt.title(description)
        plt.xticks(rotation=45, ha='right')

        # Adding text labels above the bars
        for bar, energy in zip(bars, bar_data['Area']):
            plt.annotate(f'{energy:.2f} MJ', # Text label
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), # Position
                        xytext=(0, 3),  # Offset from the top of the bar
                        textcoords='offset points',
                        ha='center', va='bottom') # Text alignment

        plt.tight_layout()
        
        description = " ".join(description.split()[:2])
        plot_name =  description + " " + self.group_name + ".png"
        path_to_plot = os.path.join(self.metrics_path, plot_name)
        plt.savefig(path_to_plot)
        # wandb.log({plot_name[:-4]: wandb.Image(path_to_plot)}, commit=True)

if __name__ == "__main__":
    main_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(main_path, "config.yaml")
    config = Config(config_path)
    group_names = config.get("wandb_metrics_extraction").get("group_names")
    for group_name in group_names:
        metrics = Metrics(config, group_name)
        metrics.extract_convergence()
        metrics.extract_system_metrics_all()
        metrics.extract_energy_consumption()
        # metrics.extract_system_metrics("system.gpu.0.powerWatts", "GPU Power Usage (W)")
        wandb.finish()

    # Call method for appropriate metrics extraction