# Projet UE-A Algorithme de recherche : Optimisation d'Hyperparamètres avec Algorithmes Génétiques - Yamine Belkhedra

Ce projet vise à explorer et optimiser les hyperparamètres d'un réseau de neurones pour résoudre le problème de classification du jeu de données MNIST. L'optimisation est réalisée à l'aide d'algorithmes génétiques, cherchant à découvrir les combinaisons d'hyperparamètres qui maximisent la performance du modèle sur un ensemble de validation.

Il est également trouvable sur https://github.com/ybelkhedra/GA-best-nn.

## Structure du Projet

```
./
|-- src/
|   |-- MNISTDataLoader.py    # Gestion du chargement des données MNIST
|   |-- models.py             # Définition des modèles de réseau de neurones
|   |-- train.py              # Fonctions d'entraînement et d'évaluation
|   |-- genetic_algorithm.py  # Implémentation de l'algorithme génétique
|-- main.py                   # Fichier principal pour exécuter l'algorithme génétique
|-- config.yaml               # Configuration de l'algorithme génétique
|-- requirements.txt          # Fichier des dépendances
```

## Guide d'utilisation

1. **Configuration**:
    - Modifiez le fichier `config.yaml` pour ajuster les paramètres de l'algorithme génétique. Il est possible de configurer la taille des batches, la taille des couches cachées initiales, la taille de la population, le nombre de génération, le nombre de parents que l'on garde entre chaque génération et le taux de mutation.

2. **Environnement**:
    - Assurez-vous d'avoir les dépendances nécessaires en installant les packages répertoriés dans `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

3. **Exécution**:
    - Lancez le script principal `main.py` pour exécuter l'algorithme génétique et optimiser les hyperparamètres du modèle.

   ```bash
   python3 main.py
   ```

4. **Résultats**:
    - Le meilleur modèle obtenu sera affiché après l'exécution du script.

## Remarques

- Ce projet utilise PyTorch pour la construction du modèle et l'entraînement du réseau de neurones.

- Les performances du modèle et la convergence de l'algorithme génétique peuvent varier en fonction des choix d'hyperparamètres initiaux et des paramètres de l'algorithme génétique.

## Expérience réalisée

J'ai réalisé un entraînement avec la configuration suivante : 
```
batch_size: 64
hidden_size: 128
population_size: 5
num_generations: 5
num_parents: 3
mutation_rate: 0.1
```

J'ai essayé d'optimiser les hyperparamètres d'un réseau de neurones pour obtenir la meilleur accuracy sur des données de validations du jeu de données MNIST.

### Modèle flexible

Est modifiable les hyperparamètres suivants :
- Le nombre et la taille des couches chachées ;
- Le taux de droupout utilisé ;
- L'utilisation de batch normalisation.

Cela est permis à l'aide d'une classe nommée "FlexibleNN" trouvable dans le fichier "./src/models.py". Cette classe a pour attributs les différents hyperparamètres cités ci-dessus pour guider l'algorithme génétique à faire des crossover et des mutations.

### Fonctionnement de l'algorithme génétique

J'ai choisi de faire fonctionner l'algorithme génétique de la manière suivante : 
- la population initiale est générée de manière aléatoire ;
- pour évaluer chaque individu, on lance un entraînement avec deux epochs (cela suffit pour avoir un bon score sur MNIST) ;
- Le score de chaque individu correspond à l'accuracy obtenue sur les données de validation ;
- Le crossover entre deux parents correspond à choisir aléatoirement les hyperparamètres du parent 1 ou du parent 2 pour avoir un mélange ;
- la mutation correspond à modifier chacun des hyperparamètres avec une probabilité de mutation_rate, et ça avec plus ou moins d'intensité.

### Résultat obtenu

```
Generation 0
Evaluating model 1/5
[1,  1000] loss: 0.497687
[2,  1000] loss: 0.274718
Evaluating model 2/5
[1,  1000] loss: 0.507006
[2,  1000] loss: 0.258550
Evaluating model 3/5
[1,  1000] loss: 0.256666
[2,  1000] loss: 0.099017
Evaluating model 4/5
[1,  1000] loss: 0.351232
[2,  1000] loss: 0.128651
Evaluating model 5/5
[1,  1000] loss: 0.400097
[2,  1000] loss: 0.187877
Ranking of the models in generation 0:
Model 0: Accuracy 0.9773
Model 1: Accuracy 0.9749
Model 2: Accuracy 0.9727
Model 3: Accuracy 0.9599
Model 4: Accuracy 0.9565
Generation 1
Evaluating model 1/5
[1,  1000] loss: 0.081493
[2,  1000] loss: 0.057058
Evaluating model 2/5
[1,  1000] loss: 0.107685
[2,  1000] loss: 0.082227
Evaluating model 3/5
[1,  1000] loss: 0.160507
[2,  1000] loss: 0.132673
Evaluating model 4/5
[1,  1000] loss: 0.306570
[2,  1000] loss: 0.100342
Evaluating model 5/5
[1,  1000] loss: 0.280419
[2,  1000] loss: 0.115872
Ranking of the models in generation 1:
Model 0: Accuracy 0.9796
Model 1: Accuracy 0.9786
Model 2: Accuracy 0.9767
Model 3: Accuracy 0.9750
Model 4: Accuracy 0.9743
Generation 2
Evaluating model 1/5
[1,  1000] loss: 0.074619
[2,  1000] loss: 0.063766
Evaluating model 2/5
[1,  1000] loss: 0.122671
[2,  1000] loss: 0.111219
Evaluating model 3/5
[1,  1000] loss: 0.049519
[2,  1000] loss: 0.036355
Evaluating model 4/5
[1,  1000] loss: 0.251940
[2,  1000] loss: 0.104090
Evaluating model 5/5
[1,  1000] loss: 0.348599
[2,  1000] loss: 0.133520
Ranking of the models in generation 2:
Model 0: Accuracy 0.9831
Model 1: Accuracy 0.9813
Model 2: Accuracy 0.9787
Model 3: Accuracy 0.9752
Model 4: Accuracy 0.9749
Generation 3
Evaluating model 1/5
[1,  1000] loss: 0.034103
[2,  1000] loss: 0.030437
Evaluating model 2/5
[1,  1000] loss: 0.056778
[2,  1000] loss: 0.054888
Evaluating model 3/5
[1,  1000] loss: 0.099931
[2,  1000] loss: 0.096599
Evaluating model 4/5
[1,  1000] loss: 0.255523
[2,  1000] loss: 0.101781
Evaluating model 5/5
[1,  1000] loss: 0.290383
[2,  1000] loss: 0.127606
Ranking of the models in generation 3:
Model 0: Accuracy 0.9825
Model 1: Accuracy 0.9812
Model 2: Accuracy 0.9807
Model 3: Accuracy 0.9749
Model 4: Accuracy 0.9701
Generation 4
Evaluating model 1/5
[1,  1000] loss: 0.047665
[2,  1000] loss: 0.042229
Evaluating model 2/5
[1,  1000] loss: 0.025600
[2,  1000] loss: 0.022378
Evaluating model 3/5
[1,  1000] loss: 0.093571
[2,  1000] loss: 0.083093
Evaluating model 4/5
[1,  1000] loss: 0.476496
[2,  1000] loss: 0.213149
Evaluating model 5/5
[1,  1000] loss: 0.347948
[2,  1000] loss: 0.161216
Ranking of the models in generation 4:
Model 0: Accuracy 0.9834
Model 1: Accuracy 0.9827
Model 2: Accuracy 0.9818
Model 3: Accuracy 0.9742
Model 4: Accuracy 0.9699
Best model accuracy: 0.9834
Best model info:
{'input_size': 784, 'hidden_sizes': [256, 256], 'output_size': 10, 'dropout_rate': 0.0013152282978935448, 'use_batch_norm': True}
```

On peut observer que l'accuracy globale entre chaque génération augmente, ce qui est rassurant. Ces changements sont néanmoins peu perceptibles car il est facile d'avoir de bon score sur MNIST. Pourtant, le modèle final retenu est celui qui a effectivement la meilleur accuracy parmi tous les modèles essayés sur toutes les générations confondues, et mieux encore il est apparu à la dernière génération.

## Ouvertures

Voici quelques points pour continuer ce projet : 
- faire alterner entre les entraînements quel sont les données de validations pour éviter un overfitting sur ceux-ci ;
- rajouter plus possibilité d'hyperparamètres en rajoutant par exemple la possibilité de mettre des couches de convolutions.