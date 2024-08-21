import numpy as np
import matplotlib.pyplot as plt


# [0, 7] interval for x and y
# Fitness function to be minimized: f(x, y) = sqrt(x^3 + 2y^4)
def fitness(x, y):
    return np.sqrt(x ** 3 + 2 * (y ** 4))


# Converts a binary string to an integer
def bin_to_int(bin_str):
    return int(bin_str, 2)


# Initializes the population with random binary strings
def initialize_population(size):
    return np.random.randint(0, 2, (size, 6))


# Roulette wheel selection
def roulette_wheel_selection(population, fitness_values, num_parents):
    total_fitness = np.sum(fitness_values)
    selection_probabilities = fitness_values / total_fitness
    cumulative_probabilities = np.cumsum(selection_probabilities)
    selected_indices = []
    for _ in range(num_parents):
        random_number = np.random.rand()
        for i, probability in enumerate(cumulative_probabilities):
            if random_number <= probability:
                selected_indices.append(i)
                break
    return population[selected_indices]


# Crossover, one-point crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    return np.concatenate((parent1[:crossover_point], parent2[crossover_point:])), \
        np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))


# Mutation, bit-flip mutation
def mutate(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        point = np.random.randint(0, len(individual))
        individual[point] = 1 - individual[point]
    return individual


# Decodes the binary strings to integers
def decode(population):
    decoded = []
    for individual in population:
        x_bin, y_bin = "".join(map(str, individual[:3])), "".join(map(str, individual[3:]))
        x, y = bin_to_int(x_bin), bin_to_int(y_bin)
        decoded.append((x, y))
    return decoded


# Genetic algorithm
def genetic_algorithm(population_size, generations, mutation_rate):
    population = initialize_population(population_size)
    best_fitness_overtime = []
    best_individual = None
    best_fitness = float('inf')

    for generation in range(generations):
        decoded_population = decode(population)
        fitness_values = np.array([fitness(x, y) for x, y in decoded_population])

        current_best_idx = np.argmin(fitness_values)
        current_best_fitness = fitness_values[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = decoded_population[current_best_idx]

        best_fitness_overtime.append(best_fitness)

        # Inversão dos valores de aptidão para seleção por roleta
        inverted_fitness_values = 1 / (1 + fitness_values)
        selected_parents = roulette_wheel_selection(population, inverted_fitness_values, population_size)
        new_population = []

        for i in range(0, population_size, 2):
            parent1_idx = np.random.randint(0, len(selected_parents))
            parent2_idx = np.random.randint(0, len(selected_parents))

            parent1 = selected_parents[parent1_idx]
            parent2 = selected_parents[parent2_idx]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = np.array(new_population[:population_size])

    return best_fitness_overtime, best_individual


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

num_runs = 100
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
plt.title("Genetic Algorithm Convergence (Average over 100 runs)")
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
