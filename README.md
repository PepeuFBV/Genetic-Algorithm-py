# Genetic Algorithm in Python
 
Genetic Algorithm is a search heuristic that is inspired by Charles Darwin’s theory of natural evolution. This algorithm reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation.

In this implementation of Genetic Algorithm, the algorithm is trying to find the solution to the following problem:

```python
f(x, y) = sqrt(x^3 + 2y^4)
```

where x and y are integers in the range [0, 7].

The algorithm will try to find the values of x and y that will minimize the value of f(x, y).

## How to run

You can run the algorithm by executing the following command:

```bash
python genetic_algorithm.py
```

## Parameters

You can adjust the following parameters in the `genetic_algorithm.py` file:

- `population_size`: The number of individuals in each generation.
- `generations`: The number of generations.
- `mutation_rate`: The probability of mutation for each individual.

## About the code

The code is divided into the following sections:

1. **Initialization**: In this section, the initial population is created with random values of x and y.
2. **Selection**: In this section, the individuals are selected for reproduction based on their fitness, using the roulette wheel selection method.
3. **Crossover**: In this section, the selected individuals are crossed over to produce offspring.
4. **Mutation**: In this section, the offspring are mutated based on the mutation rate.
5. **Survivor selection**: In this section, the new generation is created by combining the offspring generated through crossover and mutation.
6. **Evaluation**: In this section, the fitness of each individual in the new generation is evaluated.

Firstly the algorithm will run once and output the best individual found, then it will run multiple more times with plotting to display the best individual found in each generation for each parameter combination.

- It will run 100 times to get the best individual found in each generation, generating a plot for the mean of individuals fitness in each generation.

- It will run for 75 generations, changing the population size from 2 to 24 (2, 4, 6...), generating a plot for the mean of individuals fitness in each generation.

- It will run for 12 generations, changing the mutation rate from 0.1 to 0.5 (0.1, 0.15, 0.2...), generating a plot for the mean of individuals fitness in each generation.
