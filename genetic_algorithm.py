import numpy as np

# [0, 7] interval for x and y
# Fitness function to be minimized: f(x, y) = sqrt(x^3 + 2y^4)
def fitness(x, y):
    return np.sqrt(x ** 3 + 2 * (y ** 4))


# Converts a binary string to an integer
def bin_to_int(bin_str):
    return int(bin_str, 2) # converts the binary string to an integer value


# Decodes the binary strings to integers
def decode(population):
    decoded = []
    for individual in population: # decodes each individual in the population, aka the binary string to integer values
        x_bin, y_bin = "".join(map(str, individual[:3])), "".join(map(str, individual[3:])) # splits the binary string into x and y parts
        x, y = bin_to_int(x_bin), bin_to_int(y_bin) # converts the binary strings to integer values
        decoded.append((x, y))
    return decoded


# Initializes the population with random binary strings
def initialize_population(size):
    return np.random.randint(0, 2, (size, 6)) # generates a random binary string of length 6


# Roulette wheel selection, selects parents based on their fitness values
def roulette_wheel_selection(population, fitness_values, num_parents):
    total_fitness = np.sum(fitness_values) # calculates the total fitness of the population to make the selection probabilities
    selection_probabilities = fitness_values / total_fitness # calculates the selection probabilities based on the fitness values
    cumulative_probabilities = np.cumsum(selection_probabilities)  # calculates the cumulative probabilities
    selected_indices = []
    for _ in range(num_parents): # selects the parents based on the cumulative probabilities
        random_number = np.random.rand() # generates a random number between 0 and 1
        for i, probability in enumerate(cumulative_probabilities): # selects the parent based on the random number
            if random_number <= probability: # if the random number is less than the cumulative probability
                selected_indices.append(i) # selects the parent
                break
    return population[selected_indices]


# Crossover, one-point crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1)) # selects a random crossover point between 1 and the length of the individual
    return np.concatenate((parent1[:crossover_point], parent2[crossover_point:])), \
        np.concatenate((parent2[:crossover_point], parent1[crossover_point:])) # performs one-point crossover


# Mutation, bit-flip mutation
def mutate(individual, mutation_rate):
    if np.random.rand() < mutation_rate: # checks if the mutation rate is less than a random number between 0 and 1
        point = np.random.randint(0, len(individual)) # selects a random point in the individual
        individual[point] = 1 - individual[point] # flips the bit at the selected point
    return individual


# Genetic algorithm
def genetic_algorithm(population_size, generations, mutation_rate):
    population = initialize_population(population_size)
    best_fitness_overtime = []
    best_individual = None
    best_fitness = float('inf')

    for generation in range(generations): # runs the genetic algorithm for a given number of generations
        decoded_population = decode(population) # decodes the population to integer values
        fitness_values = np.array([fitness(x, y) for x, y in decoded_population]) # calculates the fitness values of the current population

        current_best_idx = np.argmin(fitness_values) # selects the index of the best individual in the population
        current_best_fitness = fitness_values[current_best_idx] # selects the fitness value of the best individual

        if current_best_fitness < best_fitness: # checks if the current best fitness is better than the previous best fitness
            best_fitness = current_best_fitness # updates the best fitness
            best_individual = decoded_population[current_best_idx] # updates the best individual

        best_fitness_overtime.append(best_fitness) # appends the best fitness to the list

        # invert the fitness values for selection
        inverted_fitness_values = 1 / (1 + fitness_values)
        selected_parents = roulette_wheel_selection(population, inverted_fitness_values, population_size) # selects the parents based on the inverted fitness values
        new_population = []

        for i in range(0, population_size, 2): # performs crossover and mutation to generate new individuals, until the population size is reached
            parent1_idx = np.random.randint(0, len(selected_parents))
            parent2_idx = np.random.randint(0, len(selected_parents))

            parent1 = selected_parents[parent1_idx]
            parent2 = selected_parents[parent2_idx]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = np.array(new_population[:population_size]) # updates the population with the new individuals

    return best_fitness_overtime, best_individual
