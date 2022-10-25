import numpy as np
import random
from individual import Individual

class ModelParameters:
    def __init__(self, n_genes, mutation_chance, pop_size):
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.mutation_chance = mutation_chance
        self.target = random.randint(0, 2**n_genes - 1)

    def random_genes(self) -> np.ndarray[int]:
        return np.random.randint(0, 2, dtype=np.int8, size=self.n_genes)
    
class BinaryBioAi:
    def __init__(self, mp: ModelParameters):
        self.mp = mp
        self.population = [Individual(mp.random_genes()) for _ in range(mp.pop_size)]
        self.best_fitness_individual = None

    def step(self):
        best_fitness = -1 * (2**self.mp.n_genes)
        for individual in self.population:
            individual.f(self.mp.target)
            if individual.fitness > best_fitness:
                self.best_fitness_individual = individual
                best_fitness = individual.fitness
                
        self.evolve()
    
    def evolve(self):
        ELITE_PERCENT = 0.80
        MAX_FAILED_MATES = 100
        MUTATION_CHANCE = 0.05

        new_population = []
        self.population.sort(key=lambda x: x.fitness)
        elites = BinaryBioAi.percent_of(self.population, ELITE_PERCENT)

        # Keep top 50% of elites
        elites_top = BinaryBioAi.percent_of(elites, 0.50)
        for individual in elites_top:
            new_population.append(individual)

        # Mate
        n_mates = (100 + random.randint(-50, 50)) - len(new_population)
        for _ in range(n_mates):
            parent1 = elites[random.randint(0, len(elites) - 1)]
            parent2 = None

            # Prevent inbreeding by making sure parent1 != parent2 OR mutating parent2 after 100 atemmpts
            for _ in range(MAX_FAILED_MATES):
                parent2 = self.population[random.randint(0, len(self.population) - 1)]
                if parent2 != parent1:
                    break
            else:
                parent2 = parent2.mutate()

            offspring = parent1.mate(parent2)
    
            new_population.append(offspring)

        # Mutate
        for individual in self.population:
            if np.random.random_sample() < MUTATION_CHANCE:
                mutated = individual.mutate()
                new_population.append(mutated)

        self.population = new_population  

    def avg_fitness(self) -> float:
        return np.average(list(map(lambda x: x.fitness, self.population)))

    def best_solution(self) -> str:
        return self.best_fitness_individual.solution

    def best_fitness(self) -> int:
        return self.best_fitness_individual.fitness

    def percent_of(a, q):
        return a[int(len(a)*(1-q)):]
