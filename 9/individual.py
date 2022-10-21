import numpy as np
import random
from typing_extensions import Self

class Individual:
    def __init__(self, genes: np.ndarray[int]):
        self.estimate = 0
        self.genes = genes
        self.solution = ''
        self.fitness = 0

    def f(self, num: int) -> str:
        genes_str = map(lambda x: str(x), self.genes)
        self.solution = ''.join(genes_str)
        self.estimate = int(self.solution, 2)
        self.fitness = int(-np.abs(self.estimate - num))
        
        return self.solution

    def mate(self, other: Self) -> Self:
        gene_tuples = zip(self.genes, other.genes)
        genes = list(map(lambda x: x[random.randint(0, 1)], gene_tuples))
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
