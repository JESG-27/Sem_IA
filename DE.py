import numpy as np
import random
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
        self.population = np.random.uniform(low=[x_min, y_min], high=[x_max , y_max], size=(self.individuals, 2))
        self.best_solutions = []
        
        minimun = np.argmin([michalewicz(self.population[i]) for i in range(self.individuals)])
        self.best_global_solution = [self.population[minimun][0], self.population[minimun][1], michalewicz(self.population[minimun])]
        
    
    def run(self):
        cont = 0
        while(cont < self.iterations):
            for ind in range(self.individuals):
                r1, r2, r3 = random.sample(range(self.individuals), 3)
                trial = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
                
                for i in range(2):
                    if random.random() < self.crossover or i == r1:
                        trial[i] = self.population[ind][i]
            
                self.borders(trial)
                if michalewicz(trial) < michalewicz(self.population[ind]):
                    self.population[ind] = trial
            
            minimun = np.argmin([michalewicz(self.population[i]) for i in range(self.individuals)])
            best_solution = [self.population[minimun][0], self.population[minimun][1], michalewicz(self.population[minimun])]
            self.best_solutions.append(best_solution)
            
            if (best_solution[2] < self.best_global_solution[2]):
                self.best_global_solution = best_solution

            cont += 1
    
    def borders(self, ind):
        if (ind[0] > x_max):
            ind[0] = x_max
        elif (ind[0] < x_min):
            ind[0] = x_min

        if (ind[1] > y_max):
            ind[0] = y_max
        elif (ind[1] < y_min):
            ind[1] = y_min

optimization = DE(50, 100)
optimization.run()
print("Best: ", optimization.best_global_solution)

# Función
figura = plt.figure(num="Michalewicz")
ejes = Axes3D(figura)
valX = np.linspace(x_min, x_max, 500)
valY = np.linspace(y_min, y_max, 500)
X, Y = np.meshgrid(valX, valY)
Z = michalewicz([X, Y])
ejes.plot_surface(X, Y, Z, cmap="viridis")
ejes.scatter(optimization.best_global_solution[0], optimization.best_global_solution[1], optimization.best_global_solution[2], marker='*', color='red', s=100)

plt.show()