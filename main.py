import yaml
import torch
from torch.utils.data import DataLoader
from src.MNISTDataLoader import MNISTDataLoader
from src.models import FlexibleNN
from src.genetic_algorithm import genetic_algorithm

def main(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    train_loader, val_loader = MNISTDataLoader(32, True, 1).load()

    input_size = 28 * 28  
    output_size = 10

    initial_model = FlexibleNN(
        input_size,
        hidden_sizes=[64, 128],
        output_size=output_size,
        dropout_rate=0.2,
        use_batch_norm=True
    )

    best_model = genetic_algorithm(
        config['population_size'],
        config['num_generations'],
        config['num_parents'],
        config['mutation_rate'],
        input_size,
        output_size,
        train_loader,
        val_loader,
        torch.nn.CrossEntropyLoss(),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )


if __name__ == "__main__":
    config_path = "config.yaml"
    main(config_path)