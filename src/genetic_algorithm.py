import torch
import random
from src.models import FlexibleNN
from src.train import evaluate_model, train

import torch
import random

def initialize_population(population_size, input_size, output_size):
    population = []
    for _ in range(population_size):
        # Exemple avec différentes valeurs d'hyperparamètres
        hidden_sizes = [random.choice([64, 128, 256]) for _ in range(random.randint(1, 3))]
        dropout_rate = random.uniform(0.0, 0.5)
        use_batch_norm = random.choice([True, False])

        model = FlexibleNN(input_size, hidden_sizes, output_size, dropout_rate, use_batch_norm)
        population.append(model)
    return population

def evaluate_population(population, train_loader, val_loader, criterion, device):
    evaluations = []
    for model in population:
        model.to(device)
        # Exemple simple d'entraînement et d'évaluation, adapte-le en fonction de tes besoins
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

        accuracy = evaluate_model(model, val_loader, device)
        evaluations.append((model, accuracy))

    return evaluations

def select_parents(evaluations, num_parents):
    sorted_evaluations = sorted(evaluations, key=lambda x: x[1], reverse=True)
    parents = [eval[0] for eval in sorted_evaluations[:num_parents]]
    return parents

def crossover(parent1, parent2):
    # Crossover des hyperparamètres
    hidden_sizes_child = []
    for size1, size2 in zip(parent1.hidden_sizes, parent2.hidden_sizes):
        hidden_sizes_child.append(random.choice([size1, size2]))

    dropout_rate_child = random.choice([parent1.dropout_rate, parent2.dropout_rate])
    use_batch_norm_child = random.choice([parent1.use_batch_norm, parent2.use_batch_norm])

    child = FlexibleNN(parent1.input_size, hidden_sizes_child, parent1.output_size, dropout_rate_child, use_batch_norm_child)
    return child

def mutate(child, mutation_rate):
    # Mutation des hyperparamètres
    mutated_child = child.clone()

    for i in range(len(mutated_child.hidden_sizes)):
        if random.random() < mutation_rate:
            mutated_child.hidden_sizes[i] = random.choice([64, 128, 256])

    if random.random() < mutation_rate:
        mutated_child.dropout_rate = random.uniform(0.0, 0.5)

    if random.random() < mutation_rate:
        mutated_child.use_batch_norm = not mutated_child.use_batch_norm

    return mutated_child

def genetic_algorithm(population_size, num_generations, num_parents, mutation_rate, input_size, output_size, train_loader, val_loader, criterion, device):
    population = initialize_population(population_size, input_size, output_size)

    for generation in range(num_generations):
        evaluations = evaluate_population(population, train_loader, val_loader, criterion, device)
        parents = select_parents(evaluations, num_parents)

        new_population = parents.copy()

        while len(new_population) < population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    best_model, _ = max(evaluations, key=lambda x: x[1])
    return best_model