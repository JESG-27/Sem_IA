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
        self.crossover = 0.2
        self.population = np.random.uniform(low=[x_min, y_min], high=[x_max , y_max], size=(self.individuals, 2))
        self.best_solutions = [],[]
        
        minimun = np.argmin([michalewicz(self.population[i]) for i in range(self.individuals)])
        self.best_global_solution = [self.population[minimun][0], self.population[minimun][1], michalewicz(self.population[minimun])]  
    
    def run(self):
        for iteration in range(self.iterations):
            for ind in range(self.individuals):
                # Mutación
                r1, r2, r3 = random.sample(range(self.individuals), 3)
                F = np.random.uniform(0.4,0.9)
                trial = self.population[r1] + F * (self.population[r2] - self.population[r3])
                
                # Cruce
                for i in range(2):
                    if random.random() < self.crossover or i == np.random.randint(0,2):
                        trial[i] = self.population[ind][i]

                # Evaluación
                if (michalewicz(trial) < michalewicz(self.population[ind]) and self.borders(trial)):
                    self.population[ind] = trial
            
            # Mejor solución
            minimun = np.argmin([michalewicz(self.population[i]) for i in range(self.individuals)])
            self.best_global_solution = [self.population[minimun][0], self.population[minimun][1], michalewicz(self.population[minimun])]
            
            self.best_solutions[0].append(self.best_global_solution[2])
            self.best_solutions[1].append(iteration)
    
    def borders(self, ind):
        if (x_min <= ind[0] and ind[0] <= x_max and y_min <= ind[1] and ind[1] <= y_max):
            return True
        else:
            return False
    
    def results(self):
        print("<------------- Resultados ------------->")
        print(f"Mejores valores: x: {self.best_global_solution[0]} y: {self.best_global_solution[1]}")
        print(f"Mejor resultado: {self.best_global_solution[2]}")
        print("<-------------------------------------->")

# Evolución Diferencial
optimization = DE(50, 100) # DE(Individuos, Iteraciones)
optimization.run()
optimization.results()

# Función
figura = plt.figure(num="Michalewicz")
ejes = Axes3D(figura)
valX = np.linspace(x_min, x_max, 500)
valY = np.linspace(y_min, y_max, 500)
X, Y = np.meshgrid(valX, valY)
Z = michalewicz([X, Y])
ejes.plot_surface(X, Y, Z, cmap="viridis")
evolution = optimization.best_global_solution
ejes.scatter(evolution[0], evolution[1], evolution[2], marker='*', color='red', s=100)

# Mejores resultados
figura = plt.figure(num="DE") 
plt.xlabel("Iteración")
plt.ylabel("Minimo encontrado")
plt.plot(optimization.best_solutions[1], optimization.best_solutions[0], color='red')

plt.show()