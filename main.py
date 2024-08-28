import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import (fitness, bin_to_int, initialize_population, roulette_wheel_selection, crossover,
                               mutate, decode, genetic_algorithm)

###########################################################################

population_size = 6
generations = 50
mutation_rate = 0.05  # 5% mutation rate

best_fitness_overtime, best_individual = genetic_algorithm(population_size, generations, mutation_rate)

print(f"Best individual:\nx = {best_individual[0]}, y = {best_individual[1]}\nFitness ="f" {fitness(best_individual[0], best_individual[1])}")

plt.plot(best_fitness_overtime)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Genetic Algorithm Convergence")
plt.show()


###########################################################################

# Running the genetic algorithm 100 times to get the average fitness over time

num_runs = 1000
all_fitness_overtime = []

for _ in range(num_runs):
    best_fitness_overtime, _ = genetic_algorithm(population_size, generations, mutation_rate)
    all_fitness_overtime.append(best_fitness_overtime)

# Calculate the average fitness over time
average_fitness_overtime = np.mean(np.array(all_fitness_overtime), axis=0)

# Plot the average fitness over time
plt.plot(average_fitness_overtime)
plt.xlabel("Generation")
plt.ylabel("Average Fitness")
plt.title("Genetic Algorithm Convergence (average over 1000 runs)")
plt.show()


###########################################################################

# Running the genetic algorithm 100 times and changing the population size from 2 to 24

generations = 75 # Increase the number of generations for better convergence
population_sizes = np.arange(2, 25, 2)
all_average_fitness_overtime = []

for population_size in population_sizes:
    all_fitness_overtime = []
    for _ in range(num_runs):
        best_fitness_overtime, _ = genetic_algorithm(population_size, generations, mutation_rate)
        all_fitness_overtime.append(best_fitness_overtime)
    average_fitness_overtime = np.mean(np.array(all_fitness_overtime), axis=0)
    all_average_fitness_overtime.append(average_fitness_overtime)

# Plot the average fitness over time for different population sizes
for i, population_size in enumerate(population_sizes):
    plt.plot(all_average_fitness_overtime[i], label=f"Population size: {population_size}")
plt.xlabel("Generation")
plt.ylabel("Average Fitness")
plt.title("Genetic Algorithm Convergence for Different Population Sizes")
plt.legend()
plt.show()


###########################################################################

# Running the genetic algorithm 100 times and changing the mutation rate from 0.1 to 0.5

generations = 12 # Decrease the number of generations for better visualization
mutation_rates = np.linspace(0.1, 0.5, 10)
all_average_fitness_overtime = []

for mutation_rate in mutation_rates:
    all_fitness_overtime = []
    for _ in range(num_runs):
        best_fitness_overtime, _ = genetic_algorithm(population_size, generations, mutation_rate)
        all_fitness_overtime.append(best_fitness_overtime)
    average_fitness_overtime = np.mean(np.array(all_fitness_overtime), axis=0)
    all_average_fitness_overtime.append(average_fitness_overtime)

# Plot the average fitness over time for different mutation rates
for i, mutation_rate in enumerate(mutation_rates):
    plt.plot(all_average_fitness_overtime[i], label=f"Mutation rate: {mutation_rate:.2f}")
plt.xlabel("Generation")
plt.ylabel("Average Fitness")
plt.title("Genetic Algorithm Convergence for Different Mutation Rates")
plt.legend()
plt.show()
