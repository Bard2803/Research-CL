import os
from config import Config
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

class Metrics():
    def __init__(self, config, group_name):
        wandb.login()
        api = wandb.Api()
        self.config = config
        project_name = config.get("wandb").get("project_name")
        self.group_name = group_name
        # wandb.init(project=project_name, group=group_name)
        self.runs = api.runs(project_name, filters = {"group": self.group_name})
        folder_name = config.get("wandb_metrics_extraction").get("folder_name")
        self.metrics_path = self.create_folder(os.path.join(folder_name, "individual_benchmarks"))
        self.benchmark_name  = group_name.split("_")[:-1][0]

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
        for run in self.runs:
            # There is discrepancy in the convergence, because the history of wandb is downloaded with  
            # different number of epochs for each experience. 
            history = run.history()
            history = pd.DataFrame(history)
            counts = [] 
            strategy_names.append(run.name)
            # Uncomment if want to print the history for each run
            # history.to_excel(os.path.join(self.metrics_path, "analysis_convergence_discrepancy", f"{run.name + str(strategy_names.count(run.name))}.xlsx"))
            for _, row in history.iterrows():
                if np.isnan(row['TrainingExperience']):
                    counter += 1
                else:
                    counter += 1
                    # Append the count to the list and reset the counter
                    counts.append(counter)
                    counter = 0
            while len(counts) < num_experiences:
                counts.append(sum(counts)/len(counts))
            df[run.name + str(strategy_names.count(run.name))] = counts
        return df
    
    def calculate_convergence(self, df):
        strategies = self.config.get("wandb_metrics_extraction").get("strategies")
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
        self.plot_bar_chart(summary)
        wandb.finish()
        print("finished extracting convergence")
        return summary

    def extract_convergence(self):
        df = self.num_epochs_per_experience()
        df = self.calculate_convergence(df)
        return df
 
    def save_metrics_xlsx(self):
        data = pd.DataFrame()
        # Fetch the logged metrics for each run
        for run in self.runs:
            history = run.history(stream="system")
            history['run_id'] = run.id
            history['strategy_name'] = run.name
            data = pd.concat([data, history], ignore_index=True)

        data.to_excel(os.path.join(self.metrics_path, f"system_metrics_{self.group_name}.xlsx"), index=False)
        data = data.infer_objects(copy=False).interpolate()
        # data = data.interpolate()
        return data
    
    def plot_bar_chart(self, summary, print_plot=False):
        # Plot
        plt.figure(figsize=(10, 5))
        bars = plt.bar(summary['strategy'], summary['mean'], yerr=summary['std'], capsize=7, color='skyblue', edgecolor='black')

        # Adding titles and labels
        plt.title(f'Mean and Standard Deviation of number of epochs to convergence for {self.benchmark_name} benchmark')
        plt.xlabel('Strategy')
        plt.ylabel('Number of epochs')

        # Adding numerical values at the top of each bar
        for bar, value in zip(bars, summary['mean']):
            plt.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 3, value),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        save_path = os.path.join(self.metrics_path, f'strategies_{self.group_name.replace("/", "-").replace(":", "-")}.png')
        plt.savefig(save_path)
        plt.close()

        # Print the summary DataFrame
        if print_plot:
            print(summary)

    def bar_plot_total_mean(self, total_mean, total_std, columns, description):
        description = "Mean " + description + " for different strategies"
        
        # Plot bar chart for total mean and standard deviation
        plt.figure(figsize=(10, 5))
        bars = plt.bar(columns, total_mean, yerr=total_std, capsize=7, color='skyblue', edgecolor='black')
        for bar, value in zip(bars, total_mean):
            plt.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 3, value),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        plt.xlabel('Strategy')
        plt.ylabel(description)
        plt.title(f"{description} for {self.benchmark_name} benchmark")
        plot_name =  description + " " + self.group_name + ".png"
        path_to_plot = os.path.join(self.metrics_path, plot_name.replace("/", "-").replace(":", "-"))
        plt.savefig(path_to_plot)
        plt.close()

    def bar_plot_runtime(self, runtime, columns):
        description = "Runtime for different strategies"
        plt.figure(figsize=(10, 5))
        # Create an array with the positions of each bar along the x-axis
        plt.bar(columns, runtime, color='orange', edgecolor='black', label='Runtime Mean (h)')
        plt.xlabel('Strategy')
        plt.ylabel(description)
        plt.title(f"{description} for {self.benchmark_name} benchmark")
        plt.legend()  # Add a legend
        plot_name =  description + " " + self.group_name + ".png"
        path_to_plot = os.path.join(self.metrics_path, plot_name.replace("/", "-").replace(":", "-"))
        plt.savefig(path_to_plot)
        plt.close()

    def plot_extract_system_metrics(self, pivot_table, pivot_table_std, description):
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
        plt.title(f'Mean {description} with Standard Deviation for Different Strategies for {self.benchmark_name} benchmark')
        plt.legend()
        

        plot_name =  description.replace("(%)", "") + " " + self.group_name + ".png"
        path_to_plot = os.path.join(self.metrics_path, plot_name.replace("/", "-").replace(":", "-"))
        plt.savefig(path_to_plot)
        plt.close()

    def extract_system_metrics(self, metric, description, interpolation_limit_system_metrics):
        data = self.save_metrics_xlsx()
        data['_runtime'] = data['_runtime'].round()
        df_mean = data.groupby(["strategy_name", "_runtime"])[metric].mean().reset_index()
        df_std = data.groupby(["strategy_name", "_runtime"])[metric].std().reset_index()
        # Pivot data to have strategies as columns
        pivot_table = df_mean.pivot(index='_runtime', columns='strategy_name', values=metric)
        pivot_table_std = df_std.pivot(index='_runtime', columns='strategy_name', values=metric)
        pivot_table = pivot_table.infer_objects(copy=False).interpolate(method='linear', limit=interpolation_limit_system_metrics)
        pivot_table_std = pivot_table_std.infer_objects(copy=False).interpolate(method='linear', limit=interpolation_limit_system_metrics)
        # pivot_table.interpolate(method='linear', inplace=True, limit=interpolation_limit_system_metrics)
        # pivot_table_std.interpolate(method='linear', inplace=True, limit=interpolation_limit_system_metrics)
        total_mean = pivot_table.mean()
        total_std = pivot_table_std.mean()
        # Cast to hours
        runtime = pivot_table.apply(lambda col: col.last_valid_index()) / (60*60)
        columns = pivot_table.columns
        self.bar_plot_total_mean(total_mean ,total_std, columns, description)
        self.bar_plot_runtime(runtime, columns)
        self.plot_extract_system_metrics(pivot_table, pivot_table_std, description)

        # TODO check why runtime is the same each iteration
        return total_mean, total_std, runtime

    def extract_system_metrics_all(self, interpolation_limit_system_metrics):
        df_total_mean = pd.DataFrame()
        df_total_std = pd.DataFrame()
        df_runtime = pd.DataFrame()
        all = {"system.gpu.0.powerWatts": "GPU Power Usage (W)", "system.gpu.0.gpu": "GPU Utilization (%)",\
                          "system.gpu.0.temp": "GPU Temperature (°C)", "system.cpu": "CPU Utilization (%)",\
                            "system.memory": "System Memory Utilization (%)"}
        for metric, description in all.items():
            total_mean, total_std, runtime = self.extract_system_metrics(metric, description, interpolation_limit_system_metrics)
            total_mean.name, total_std.name = description, description
            df_total_mean = pd.concat([df_total_mean, total_mean], axis=1)
            df_total_std = pd.concat([df_total_std, total_std], axis=1)
            # Runtime shouldnt be concatenated, because it is the same each iteration above
            # Therefore the last runtime can be returned and it is OK
        df_runtime = runtime
        return df_total_mean, df_total_std, df_runtime


    def extract_energy_consumption(self, interpolation_limit_energy_consumption):
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
        # data = data.infer_objects(copy=False).interpolate(inplace=True, limit=interpolation_limit_energy_consumption)
        pivot_table.interpolate(inplace=True, limit=interpolation_limit_energy_consumption)

        pivot_table_std = df_mean.pivot(index='mean_runtime', columns='strategy_name', values='std_metric')
        # data = data.infer_objects(copy=False).interpolate(inplace=True, limit=interpolation_limit_energy_consumption)
        pivot_table_std.interpolate(inplace=True, limit=interpolation_limit_energy_consumption)

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
            division_fac = 1e3
            energy = np.trapz(means, x) / division_fac
            std_dev = np.trapz(stds, x_stds) / division_fac
            
            # Append area and standard deviation values to the lists
            energy_list.append(energy)
            std_devs.append(std_dev)

        # Create a DataFrame to hold the data for the bar plot
        energy_consumption = pd.DataFrame({
            'Strategy': pivot_table.columns,
            'Area': energy_list,
            'Std Dev': std_devs
        })
        self.plot_energy_consumption(energy_consumption, division_fac)
        return energy_consumption


    def plot_energy_consumption(self, bar_data, division_fac, additional_desc=''):
        # Sort bar_data in descending order based on 'Area'
        bar_data = bar_data.sort_values(by='Area', ascending=False)

        # Plotting
        plt.figure(figsize=(10, 6))

        # Bar plot for energy_list and standard deviations
        bars = plt.bar(bar_data['Strategy'], bar_data['Area'], capsize=5)

        # Bar plot for energy_list and standard deviations
        plt.bar(bar_data['Strategy'], bar_data['Area'], capsize=5)
        
        unit = {1: "J", 1e3: "kJ", 1e6: "MJ", 1e9: "GJ"}
        # Adding labels and title
        plt.xlabel('Strategy')
        plt.ylabel(f'Energy ({unit[division_fac]})')
        description = f'GPU energy used for training for different strategies {additional_desc}'
        if additional_desc:
            plt.title(description + additional_desc)
            description = " ".join(description.split()[:2])
            plot_name =  description + " " + additional_desc + ".png"  
        else:   
            plt.title(f"{description} for {self.benchmark_name} benchmark")
            description = " ".join(description.split()[:2])
            plot_name =  description + " " + self.group_name + ".png"     
        plt.xticks(rotation=45, ha='right')

        # Adding text labels above the bars
        for bar, energy in zip(bars, bar_data['Area']):
            plt.annotate(f'{energy:.2f} {unit[division_fac]}', # Text label
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), # Position
                        xytext=(0, 3),  # Offset from the top of the bar
                        textcoords='offset points',
                        ha='center', va='bottom') # Text alignment

        plt.tight_layout()
        
        
        path_to_plot = os.path.join(self.metrics_path, plot_name.replace("/", "-").replace(":", "-").replace(" ", "_"))
        plt.savefig(path_to_plot)
        plt.close()

        
        # wandb.log({plot_name[:-4]: wandb.Image(path_to_plot)}, commit=True)

class Metrics_across_all_benchmarks(Metrics):
    def __init__(self, config):
        self.df_convergence = pd.DataFrame()
        self.df_system_metrics = pd.DataFrame()
        self.df_system_metrics_std = pd.DataFrame()
        self.df_runtime = pd.DataFrame()
        self.df_energy_consumption = pd.DataFrame()
        folder_name = config.get("wandb_metrics_extraction").get("folder_name")
        self.metrics_path = self.create_folder(os.path.join(folder_name, "across_all_benchmarks"))

    def plot_bar_chart(self, summary, print_plot=False):
        plt.figure(figsize=(10, 5)) 
        bars = plt.bar(summary['strategy'], summary['mean'], yerr=summary['std'], capsize=7, color='skyblue', edgecolor='black')
        # Adding titles and labels
        plt.title(f'Mean and Standard Deviation of number of epochs to convergence across all benchmarks')
        plt.xlabel('Strategy')
        plt.ylabel('Number of epochs')

        # Adding numerical values at the top of each bar
        for bar, value in zip(bars, summary['mean']):
            plt.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 3, value),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        date = datetime.now().strftime("%Y%m%d")
        save_path = os.path.join(self.metrics_path, f'convergence_across_all_benchmarks{date}.png')
        plt.savefig(save_path)
        plt.close()

        # Print the summary DataFrame
        if print_plot:
            print(summary)

    def bar_plot_total_mean(self, total_mean, total_std, columns, description):
        description = "Mean " + description 
        
        # Plot bar chart for total mean and standard deviation
        plt.figure(figsize=(10, 5))
        bars = plt.bar(columns, total_mean, yerr=total_std, capsize=7, color='skyblue', edgecolor='black')
        for bar, value in zip(bars, total_mean):
            plt.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 3, value),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        plt.xlabel('Strategy')
        plt.ylabel(description)
        plt.title(f"{description} for different strategies")
        plot_name =  description + ".png"
        path_to_plot = os.path.join(self.metrics_path, plot_name.replace("/", "-").replace(":", "-").replace(" ", "_"))
        plt.savefig(path_to_plot)
        plt.close()

    def bar_plot_runtime(self, runtimes):
        description = "Runtime for different strategies across all benchmarks"
        plt.figure(figsize=(10, 5))
        # Create an array with the positions of each bar along the x-axis
        plt.bar(runtimes.index, runtimes, color='orange', edgecolor='black', label='Runtime Mean (h)')
        plt.xlabel('Strategy')
        plt.ylabel(description)
        plt.title(f"{description} for different strategies")
        plt.legend()  # Add a legend
        plot_name =  description + ".png"
        path_to_plot = os.path.join(self.metrics_path, plot_name.replace("/", "-").replace(":", "-").replace(" ", "_"))
        plt.savefig(path_to_plot)
        plt.close()

    def concat_convergence(self, df):
        self.df_convergence = pd.concat([self.df_convergence, df.reset_index()], ignore_index=True)
    
    def concat_system_metrics(self, total_mean, total_std, runtime):
        self.df_system_metrics = pd.concat([self.df_system_metrics, total_mean.reset_index()], ignore_index=True)
        self.df_system_metrics_std = pd.concat([self.df_system_metrics_std, total_std.reset_index()], ignore_index=True)
        self.df_runtime = pd.concat([self.df_runtime, runtime.reset_index()], ignore_index=True)
    
    def concat_energy_consumption(self, df):
        self.df_energy_consumption = pd.concat([self.df_energy_consumption, df], ignore_index=True)
    
    def extract_convergence(self):
        df = self.df_convergence.groupby("strategy").mean().reset_index()
        self.plot_bar_chart(df)

    def extract_system_metrics(self):
        total_mean = self.df_system_metrics.groupby("index").mean()
        total_std = self.df_system_metrics_std.groupby("index").mean()
        bars = total_mean.index
        for (_, mean), (_, std) in zip(total_mean.items(), total_std.items()):
            description = mean.name + " across different benchmarks"
            self.bar_plot_total_mean(mean, std, bars, description)
        # TODO Implement standard deviation
        runtimes = self.df_runtime.groupby("strategy_name").mean()
        runtimes = runtimes.squeeze()
        self.bar_plot_runtime(runtimes)

    def extract_energy_consumption(self):
        energy_consumption= self.df_energy_consumption.groupby("Strategy").mean().reset_index()
        self.plot_energy_consumption(energy_consumption, 1e6, " across different benchmarks")

if __name__ == "__main__":
    main_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(main_path, "config.yaml")
    config = Config(config_path)
    metrics_aab = Metrics_across_all_benchmarks(config)
    group_names = config.get("wandb_metrics_extraction").get("group_names")
    interpolation_limit_system_metrics = 10
    interpolation_limit_energy_consumption = 10
    for group_name in group_names:
        metrics = Metrics(config, group_name)
        df = metrics.extract_convergence()
        metrics_aab.concat_convergence(df)
        total_mean, total_std, runtime = metrics.extract_system_metrics_all(interpolation_limit_system_metrics)
        metrics_aab.concat_system_metrics(total_mean, total_std, runtime)
        df = metrics.extract_energy_consumption(interpolation_limit_energy_consumption)
        metrics_aab.concat_energy_consumption(df)
        wandb.finish()
    metrics_aab.extract_convergence()
    metrics_aab.extract_system_metrics()
    metrics_aab.extract_energy_consumption()