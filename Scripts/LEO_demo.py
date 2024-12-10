import numpy as np
from sklearn.neural_network import MLPRegressor
import time
import matplotlib.pyplot as plt

# OPTIMIZATION PROBLEM
# Find global minimum of a given function

# Set main parameters for the LEO algorithm
population_size = 100  # Size of the population
dimensions = 10  # Dimensionality of each individual
generations = 50  # Number of generations (iterations)
archive_size = 100  # Maximum number of solution pairs to store in the archive
alpha = 0.5  # Weighting factor for learning-aided mutation
beta = 0.9  # Probability threshold for learning-aided crossover

# Probability of using the learning-aided operator (min and max) * 10
learning_probability_min = 0
learning_probability_max = 11

# Initialize the ANN model, with one hidden layer of size 3*dimensions
ann = MLPRegressor(hidden_layer_sizes=(dimensions * 3,),
                   activation='logistic', max_iter=1, warm_start=True)


# Define a fitness function (example: sum of squares function)
def fitness_function(x):
    return np.sum(x ** 2)  # Fitness: sum of squared values of the vector x


# Initialize the population randomly within a given range
population = np.random.uniform(-100, 100, (population_size, dimensions))

# Set each individual as their own initial personal best
personal_best = population.copy()
personal_best_fitness = np.array([fitness_function(ind) for ind in population])

# Determine the global best individual from initial personal bests
global_best = personal_best[np.argmin(personal_best_fitness)]

# Archive to store successful solution pairs for training the ANN
archive = []

# Lists to observe the results
global_best_list = []
probability_list = []
fitness_list = []
optimization_duration = []

# Iterate through the learning probabilities
for i in range(learning_probability_min, learning_probability_max):
    learning_probability = i / 10

    probability_list.append(learning_probability)  # Append probability

    # Main optimization loop
    start_time = time.time()  # start timer
    for generation in range(generations):
        new_population = []  # Container for new generation of solutions

        # Iterate over each individual in the population
        for i, individual in enumerate(population):

            # Basic PSO velocity and position update (without inertia weight and acceleration)
            velocity = np.random.rand(dimensions) * (personal_best[i] - individual) \
                       + np.random.rand(dimensions) * (global_best - individual)
            new_solution = individual + velocity  # New solution based on PSO movement

            # Check if the learning-aided operator should be applied based on probability
            if generation > 0 and np.random.rand() < learning_probability:
                # ANN predicts an improved solution based on the current individual
                predicted_solution = ann.predict(individual.reshape(1, -1))[0]

                # Learning-aided mutation: modifies predicted solution to enhance exploration
                new_solution = predicted_solution + alpha * (personal_best[np.random.randint(population_size)] \
                                                             - personal_best[np.random.randint(population_size)])
                # Learning-aided crossover: combines ANN output with current individual with probability beta
                new_solution = np.where(np.random.rand(dimensions) < beta, new_solution, individual)

            new_population.append(new_solution)  # Append modified or original solution to new population

            # Calculate fitness of the new solution
            fitness = fitness_function(new_solution)

            # Update personal and global bests if new solution has better fitness
            if fitness < personal_best_fitness[i]:
                personal_best[i] = new_solution
                personal_best_fitness[i] = fitness
            if fitness < fitness_function(global_best):
                global_best = new_solution  # Update global best if current solution is better

            # Collect successful solution pairs for ANN training if fitness improved
            if fitness < fitness_function(individual):
                archive.append((individual, new_solution))  # Store as (old_solution, new_solution) pair
                if len(archive) > archive_size:
                    archive.pop(0)  # Maintain archive size limit by removing oldest pairs

        # Update the ANN model if successful solution pairs exist in the archive
        if archive:
            X_train, y_train = zip(*archive)  # Separate old and new solutions for training
            ann.fit(np.array(X_train), np.array(y_train))  # Train ANN on collected successful pairs

        # Replace the population with the new generation
        population = np.array(new_population)

    end_time = time.time()  # end timer

    # Append results
    optimization_duration.append((end_time-start_time))
    global_best_list.append(global_best)
    fitness_list.append(fitness_function(global_best))

# Output the solutions
for i in range(learning_probability_min, learning_probability_max):
    print("======================================\n")
    print("Probability:\n", probability_list[i])
    print("Optimal Solution:\n", global_best_list[i])
    print("Optimal Fitness:\n", fitness_list[i])
    print("Optimization duration (sec):\n", optimization_duration[i])
    print("\n")

# Create the figure and add a super title
plt.figure(figsize=(8, 10))
plt.suptitle(r'Objective Function: $f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2$', 
             fontsize=14, y=0.95)
# Plotting the results (x-learning probability, y-optimization duration)
plt.subplot(2, 1, 1)  # Keep 2 rows since we're using a single figure title
plt.plot(probability_list, optimization_duration)
plt.xlabel('x - learning probability')
plt.ylabel('y - optimization duration')
plt.title('LP / DURATION')
plt.grid()

# Plotting the results (x-learning probability, y-fitness value)
plt.subplot(2, 1, 2)  # Second row
plt.plot(probability_list, fitness_list)
plt.xlabel('x - learning probability')
plt.ylabel('y - fitness value')
plt.title('LP / FITNESS')
plt.grid()

# Show all plots
plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout to leave space for the super title
plt.show()
