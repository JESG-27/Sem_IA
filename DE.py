import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x_max = 0
x_min = -10
y_max = 0
y_min = -10

# Función a optimizar
def michalewicz(X):
    resultado = 0
    for i in range(len(X)):
        resultado -= np.sin(X[i]) * np.sin(((i + 1) * X[i]**2) / np.pi)**(2 * 10) # m=10
    return resultado


class DE:
    def __init__(self, individuals, iterations):
        self.individuals = individuals
        self.iterations = iterations
        self.F = np.random.uniform(0.4,0.9)
        self.crossover = np.random.uniform(0.1, 1)
        self.population = []
        
        solutions = []
        for i in range(self.individuals):
            x = np.random.uniform(x_min,x_max)
            y = np.random.uniform(y_min,y_max)
            z = michalewicz([x,y])
            self.population.append([x,y])
            solutions.append(z)
        
        index = np.argmin(solutions)    
        self.best_solution = []
    
    def run(self):
        cont = 0
        while(cont < self.iterations):
            for ind in range(self.individuals):
                r1, r2, r3 = np.random.random_integers(0, self.individuals-1), np.random.random_integers(0, self.individuals-1), np.random.random_integers(0, self.individuals-1)
                v = self.population[r1] + self.F * (self.population[0][r2] - self.population[r3])

optimization = DE(5, 10)
optimization.run()