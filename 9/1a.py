import numpy as np
import random
from typing_extensions import Self

DEFAULT_GENES = 8
MUTATION_CHANCE = 0.05
TARGET = random.randint(0, 2**DEFAULT_GENES - 1)

def percent_of(a, q):
    return a[int(len(a)*(1-q)):]

class Individual:
    def __init__(self, genes: np.ndarray[int]):
        self.estimate = 0
        self.genes = genes
        self.solution = ''
        if self.genes is None:
            self.genes = np.random.randint(0, 2, dtype=np.int8, size=DEFAULT_GENES)
        self.fitness = 0

    def f(self, num: int) -> str:
        genes_str = map(lambda x: str(x), self.genes)
        self.solution = ''.join(genes_str)
        self.estimate = int(self.solution, 2)
        self.fitness = int(-np.abs(self.estimate - num))
        
        return self.solution

    def mate(self, other: Self) -> Self:
        gene_tuples = zip(self.genes, other.genes)
        genes = list(map(np.random.choice, gene_tuples))
        return Individual(genes)

    def clone(self) -> Self:
        return Individual(self.genes)

    def mutate(self) -> Self:
        m = self.clone()
        flip = random.randint(0, len(self.genes) - 1)
        m.genes[flip] ^= 1
        return m

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, Individual):
            return False
        return np.array_equal(self.genes, obj.genes)


    def __str__(self) -> str:
        gs = map(lambda x: str(x), self.genes)
        return ''.join(gs)

class BinaryBioAi:
    def __init__(self, pop_size: int, target: int):
        self.pop_size = pop_size
        self.population = [Individual(None) for _ in range(pop_size)]
        self.target = target
        self.best_fitness_individual = None

    def step(self):
        best_fitness = -1 * (2**DEFAULT_GENES) #2**8 -> 256
        for individual in self.population:
            individual.f(self.target)
            if individual.fitness > best_fitness:
                self.best_fitness_individual = individual
                best_fitness = individual.fitness
                
        self.evolve()
        
    def evolve(self):
        ELITE_PERCENT = 0.40
        MAX_FAILED_MATES = 100

        new_population = []
        self.population.sort(key=lambda x: x.fitness)
        elites = percent_of(self.population, ELITE_PERCENT)

        # Keep top 50% of elites
        elites_top20 = percent_of(elites, 0.50)
        for individual in elites_top20:
            print(f"Keeping {individual}")
            new_population.append(individual)

        # Mate
        while len(new_population) < 2*len(self.population):
            parent1 = np.random.choice(elites)
            parent2 = np.random.choice(self.population)
            offspring = parent1.mate(parent2)
            print(f"Mating {parent1} with {parent2} -> {offspring}")
    
            new_population.append(offspring)

        # Mutate
        for individual in self.population:
            if np.random.random_sample() < MUTATION_CHANCE:
                mutated = individual.mutate()
                print(f"Mutating {individual} -> {mutated}")
                new_population.append(mutated)

        self.population = new_population  

    def avg_fitness(self) -> float:
        return np.average(list(map(lambda x: x.fitness, self.population)))

    def best_solution(self) -> str:
        return self.best_fitness_individual.solution

    def best_fitness(self) -> int:
        return self.best_fitness_individual.fitness

ai = BinaryBioAi(pop_size=10, target=TARGET)

done = False
generation = 0
while not done:
    ai.step()

    avg_fitness = ai.avg_fitness()
    best_fitness = ai.best_fitness()
    best_solution = ai.best_solution()

    print(f"Generation: #{generation+1:07}: Target: {bin(TARGET)[2:]} Best solution: {best_solution} Avg. fit: {avg_fitness:6.2f} Best fit: {best_fitness:04}, Population: {len(ai.population):05}")

    if best_fitness == 0:
        done = True
    
    generation += 1

print("\nDone!")
