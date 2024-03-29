# Research-CL
### Resource Efficiency in Continual Learning

Repo containing code for the research in the field of Continual Learning.
Different CL strategies are compared in case of their resource efficiency.

## HOW TO RUN

### 1. Training

#### Locally
1. Build environment:
    ```pip install -r requirements.txt```
2. Check config.yaml for your training configuration
3. Launch training script
    ```python main.py```

#### Docker
1. Go to the project folder.
2. Check config.yaml for your training configuration
3. Build docker image:
    ```docker build -t your_image_name .```
4. Run docker container on the image. Connecting to your wandb account using API key:
    ```run -e WANDB_API_KEY=your_api_key_here your_image_name```

### 2. Metrics extraction

1. Build environment:
    ```pip install -r requirements.txt```
2. Go to config.yaml and change 'wandb_metrics_extraction.group_names' for you specific group names in wandb
3. Launch metrics script
    ```extract_metrics.py```



