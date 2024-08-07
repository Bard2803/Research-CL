import os
from config import Config
import wandb
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' for non-interactive plotting
import matplotlib.pyplot as plt
# Now you can create plots, save them to files, etc., without requiring a display.
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
    
    def get_accuracies(self):
        df= pd.DataFrame()
        # Fetch the logged metrics for each run
        for run in self.runs:
            history = run.history()
            history = pd.DataFrame(history)
            data = pd.DataFrame()
            # Test accuracy after final experience (task)
            try:
                data['test_accuracy'] =  history['Top1_Acc_Stream/eval_phase/test_stream/Task009'].dropna()
            except KeyError:
                data['test_accuracy'] =  history['Top1_Acc_Stream/eval_phase/test_stream/Task000'].dropna()
            data['strategy_name'] = run.name
            data = data.groupby("strategy_name").tail(1)
            df = pd.concat([df, data], ignore_index=True)
        stats_df = df.groupby('strategy_name')['test_accuracy'].agg(['mean', 'std']).reset_index()
        stats_df.columns = ['strategy', 'mean', 'std']
        stats_df = stats_df.sort_values('mean', ascending=False).round(2)
        return stats_df
    
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
        summary = summary.sort_values(by='mean', ascending=False)
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
    
    def extract_metrics(self, project_name, metric, description):
        # Login to the wandb
        wandb.login()
        # Extract the metrics from WandB after all the runs
        api = wandb.Api()
        runs = api.runs(project_name)
        # Initialize empty DataFrame
        data = pd.DataFrame()

        # Fetch the logged metrics for each run
        for run in runs:
            history = run.history() # 'default or system'
            history['run_id'] = run.id
            history['strategy_name'] = run.name
            data = pd.concat([data, history], ignore_index=True)

        data.to_excel("metrics.xlsx", index=False)


        # Create a helper column to detect changes in strategy_name
        data['index_run'] = data.groupby("run_id").cumcount()
        grouped = data.groupby(["strategy_name", "run_id"])
        data[metric] = grouped[metric].apply(lambda group: group.interpolate())
        data[metric] = data[metric]*100

        df_mean = data.groupby(["strategy_name", "index_run"])[metric, "_step"].mean().reset_index()
        df_std = data.groupby(["strategy_name", "index_run"])[metric].std().reset_index()
        df_std['_step'] = data.groupby(["strategy_name", "index_run"])["_step"].mean().reset_index()["_step"]

        # Pivot data to have strategies as columns
        pivot_table = df_mean.pivot(index='_step', columns='strategy_name', values=metric)
        pivot_table_std = df_std.pivot(index='_step', columns='strategy_name', values=metric)

        # linear for test set accuracy, akima for train set accuracy
        pivot_table.interpolate(method='linear', inplace=True, limit=10)
        pivot_table_std.interpolate(method='linear', inplace=True, limit=10)

        plt.figure(figsize=(12, 6))

        # Define colors for each strategy
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # Loop through each strategy
        for i, strategy in enumerate(pivot_table.columns):
            means = pivot_table[strategy] 
            stds = pivot_table_std[strategy] 
            x = pivot_table.index 
            # Plot means for this strategy with customizations
            plt.plot(x, means,
                    label=strategy,
                    linewidth=2,
                    color=colors[i % len(colors)])
            
            # Add variance as shadowed region
            plt.fill_between(x, means - stds, means + stds,
                            color=colors[i % len(colors)], alpha=0.2)

        # Adding labels and title
        plt.xlabel('iterations')
        plt.ylabel(f'Mean {description}')
        plt.title(f'Mean {description} with Standard Deviation for Different Strategies')
        plt.legend()

        # Show plot
        plt.show()

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
        df_runtime = runtime.to_frame().rename(columns={0: "runtime"})
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
            'StdDev': std_devs
        })
        self.plot_energy_consumption(energy_consumption, division_fac)
        return energy_consumption

    def plot_energy_consumption(self, bar_data, division_fac, additional_desc=''):
        # Sort bar_data in descending order based on 'Area'
        bar_data = bar_data.sort_values(by='Area', ascending=False)

        # Plotting
        plt.figure(figsize=(10, 6))

        # Bar plot for energy_list with standard deviations
        bars = plt.bar(bar_data['Strategy'], bar_data['Area'], yerr=bar_data['StdDev'], capsize=5)  # Added 'yerr' for standard deviation

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
        self.df_accuracies = pd.DataFrame()
        self.df_convergence = pd.DataFrame()
        self.df_system_metrics = pd.DataFrame()
        self.df_system_metrics_std = pd.DataFrame()
        self.df_runtime = pd.DataFrame()
        self.df_energy_consumption = pd.DataFrame()
        folder_name = config.get("wandb_metrics_extraction").get("folder_name")
        self.metrics_path = self.create_folder(os.path.join(folder_name, "across_all_benchmarks"))

    def plot_bar_chart(self, summary, desc, print_plot=False):
        plt.figure(figsize=(10, 5)) 
        bars = plt.bar(summary['strategy'], summary['mean'], yerr=summary['std'], capsize=7, color='skyblue', edgecolor='black')
        # Adding titles and labels
        plt.title(f'{desc} across all benchmarks for different strategies')
        plt.xlabel('Strategy')
        plt.ylabel('Number of epochs')

        # Adding numerical values at the top of each bar
        for bar, value in zip(bars, summary['mean']):
            plt.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 3, value),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        date = datetime.now().strftime("%Y%m%d")
        desc = desc.replace(" ", "_")
        save_path = os.path.join(self.metrics_path, f'{desc}_across_all_benchmarks{date}.png')
        plt.savefig(save_path)
        plt.close()

        # Print the summary DataFrame
        if print_plot:
            print(summary)

    def bar_plot_total_mean(self, df_metrics):
        df_metrics = df_metrics.sort_values(by='mean', ascending=False)
        description = "Mean " + df_metrics['description'].iloc[0]
        
        # Plot bar chart for total mean and standard deviation
        plt.figure(figsize=(10, 5))
        bars = plt.bar(df_metrics['bars'], df_metrics['mean'], yerr=df_metrics['std'], capsize=7, color='skyblue', edgecolor='black')
        for bar, value in zip(bars, df_metrics['mean']):
            plt.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 3, value),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        plt.xlabel('Strategy')
        plt.ylabel(description)
        plt.title(f"{description} for different strategies")
        plot_name =  description + ".png"
        path_to_plot = os.path.join(self.metrics_path, plot_name.replace("/", "-").replace(":", "-").replace(" ", "_"))
        plt.savefig(path_to_plot)
        plt.close()

    def bar_plot_runtime(self, runtimes_stats):
        description = "Runtime for different strategies across all benchmarks"
        plt.figure(figsize=(10, 5))
        runtimes_stats = runtimes_stats.sort_values(by='mean_runtime', ascending=False)
        bars = plt.bar(runtimes_stats.index, runtimes_stats['mean_runtime'], 
                yerr=runtimes_stats['std_runtime'], capsize=5)
        plt.xlabel('Strategy')
        plt.ylabel('Runtime (s)')
        plt.title(f"{description} for different strategies")

        # Annotate numerical values on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plot_name = description + ".png"
        path_to_plot = os.path.join(self.metrics_path, plot_name.replace("/", "-").replace(":", "-").replace(" ", "_"))
        plt.savefig(path_to_plot)
        plt.close()

    def concat_accuracies(self, df):
        self.df_accuracies = pd.concat([self.df_accuracies, df], ignore_index=True)

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
        df = df.sort_values('mean', ascending=False).round(2)
        self.plot_bar_chart(df, 'Mean number of epochs to convergence')

    def extract_accuracies(self):
        df = self.df_accuracies.groupby('strategy').mean().reset_index()
        df = df.sort_values('mean', ascending=False).round(2)
        self.plot_bar_chart(df, 'Mean accuracy')

    def extract_system_metrics(self):
        # Calculate mean and standard deviation for system metrics
        total_mean = self.df_system_metrics.groupby("index").mean()
        total_std = self.df_system_metrics.groupby("index").std()

        # Iterate over each metric in the system metrics DataFrame
        for metric in total_mean.columns:
            mean_series = total_mean[metric]
            std_series = total_std[metric]
            description = f"{metric} across all benchmarks"

            # Create a DataFrame for each metric with its mean, std, and description
            df_metrics = pd.DataFrame({
                'bars': mean_series.index,
                'mean': mean_series.values,
                'std': std_series.values,
                'description': [description] * len(mean_series)
            })

            # Call the plotting function for each metric
            self.bar_plot_total_mean(df_metrics)

        # Calculate mean and standard deviation for runtime metrics
        runtimes_stats = pd.DataFrame()
        runtimes_stats['mean_runtime'] = self.df_runtime.groupby("strategy_name").mean()['runtime']
        runtimes_stats['std_runtime'] = self.df_runtime.groupby("strategy_name").std()['runtime']

        # Pass the DataFrame with mean and std runtime to the plotting function
        self.bar_plot_runtime(runtimes_stats)

    def extract_energy_consumption(self):
        energy_consumption= self.df_energy_consumption.groupby("Strategy").mean().reset_index()
        self.plot_energy_consumption(energy_consumption, 1e6, " across all benchmarks")

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
        df = metrics.get_accuracies()
        metrics_aab.concat_accuracies(df)
        df = metrics.extract_convergence()
        metrics_aab.concat_convergence(df)
        total_mean, total_std, runtime = metrics.extract_system_metrics_all(interpolation_limit_system_metrics)
        metrics_aab.concat_system_metrics(total_mean, total_std, runtime)
        df = metrics.extract_energy_consumption(interpolation_limit_energy_consumption)
        metrics_aab.concat_energy_consumption(df)
        df = metrics
        wandb.finish()
    metrics_aab.extract_accuracies()
    metrics_aab.extract_convergence()
    metrics_aab.extract_system_metrics()
    metrics_aab.extract_energy_consumption()