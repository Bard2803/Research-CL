from extract_metrics import Metrics_across_all_benchmarks, Metrics
import pytest
import os
from config import Config

@pytest.fixture
def metrics():
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(main_path, "config.yaml")
    config = Config(config_path)
    group_name = config.get("wandb_metrics_extraction").get("group_names")[0]
    return Metrics(config, group_name)


    
def test_acc_extractio():
    metrics.extract_metrics_for_acc_xlsx()