import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="config")
def train(config: DictConfig) -> None:
    from src.train import train
    from src.utils import utils

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    train()
