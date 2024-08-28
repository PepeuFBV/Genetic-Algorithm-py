# Genetic Algorithm in Python
 
## How to run

To run the code, you will need to have Python installed on your system. You can download Python from the official website [here](https://www.python.org/downloads/). The code is written in Python 3.11.0, so it is recommended to use this version of Python to run the code.

To install the required dependencies, you can run the following command:

```bash
pip install -r requirements.txt
```

To run the code, you can use the following command:

```bash
python genetic_algorithm.py
```

## The Genetic Algorithm

Genetic Algorithm is a search heuristic that is inspired by Charles Darwin’s theory of natural evolution. This algorithm reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation.

In this implementation of Genetic Algorithm, the algorithm is trying to find the solution to the following problem:

```python
f(x, y) = sqrt(x^3 + 2y^4)
```

When x and y are integers in the defined range of [0, 7].

The algorithm will try to find the values of x and y that will minimize the value of f(x, y). The expected output for the following function is x = 0 and y = 0, which will result in f(x, y) = 0.

## Parameters

You can adjust the following parameters in the `main.py` file:

- `population_size`: The number of individuals in each generation.
- `generations`: The number of generations.
- `mutation_rate`: The probability of mutation for each individual.

## About the code

The code is divided into the following sections:

1. **Initialization**: The initial population is created with random values of x and y.
2. **Selection**: The individuals are selected for reproduction based on their fitness, using the Roulette Wheel selection method.
3. **Crossover**: The selected parents (in the previous step) are used to create offspring through crossover, the population size will remain the same.
4. **Mutation**: The offspring are chosen and mutated based on the mutation rate.
5. **Replacement**: The new generation is created by replacing the old generation with the new population.
6. **Evaluation**: In this section, the fitness of each individual in the new generation is evaluated to be further analyzed.

### Plotting and Analysis

Firstly the algorithm will run once and output the best individual found, then it will run multiple more times with plotting to display the best individual found in each generation for each parameter combination.

- It will run 1000 times to get the best individual found in each generation, generating a plot for the mean of individuals fitness in each generation.

- It will run for 75 generations, changing the population size from 2 to 24 (2, 4, 6...), generating a plot for the mean of individuals fitness in each generation.

- It will run for 12 generations, changing the mutation rate from 0.1 to 0.5 (0.1, 0.15, 0.2...), generating a plot for the mean of individuals fitness in each generation.
