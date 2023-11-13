import numpy as np
from random import random
import matplotlib.pyplot as plt

x_min = -10
x_max = 0
y_min = -10
y_max = 0

# Función a optimizar
def michalewicz(X):
    resultado = 0
    for i in range(len(X)):
        resultado -= np.sin(X[i]) * np.sin(((i + 1) * X[i]**2) / np.pi)**(2 * 10) # m=10
    return resultado

class beeHive:
    def __init__(self, iterations, bees, selection):
        self.iterations = iterations
        self.bees = [[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for i in range(bees)]
        self.bees_cont = [0 for i in range(bees)]
        self.selection = selection
        self.limit = bees//4
        self.evolution = [],[]
        minimum = np.argmin([michalewicz(self.bees[i]) for i in range(len(self.bees))])
        self.best_solution = [self.bees[minimum][0], self.bees[minimum][1], michalewicz(self.bees[minimum])]

    def run(self):
        for it in range(self.iterations):
            # Obreras
            for bee in range(len(self.bees)):
                new_bee = [self.bees[bee][0] + np.random.uniform(-1, 1), self.bees[bee][1] + np.random.uniform(-1, 1)]

                if michalewicz(new_bee) < michalewicz(self.bees[bee]):
                    self.bees[bee] = new_bee
                    self.bees_cont[bee] = 0
                else:
                    self.bees_cont[bee] += 1
            
            # Espectadoras
            for bee in range(len(self.bees)):
                if random() < self.selection:
                    minimum = np.argmin([michalewicz(self.bees[i]) for i in range(len(self.bees))])
                    new_bee = [self.bees[minimum][0] + np.random.uniform(-1, 1), self.bees[minimum][1] + np.random.uniform(-1, 1)]
                    
                    if michalewicz(new_bee) < michalewicz(self.bees[bee]):
                        self.bees[bee] = new_bee
                        self.bees_cont[bee] = 0
                    else:
                        self.bees_cont[bee] += 1
            
            # Exploradoras
            for bee in range(len(self.bees)):
                if self.bees_cont[bee] > self.limit:
                    new_bee = [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]
                    self.bees[bee] = new_bee
                    self.bees_cont[bee] = 0
        
            minimum = np.argmin([michalewicz(self.bees[i]) for i in range(len(self.bees))])
            minimum_result = michalewicz(self.bees[minimum])
            if (minimum_result < self.best_solution[2]):
                self.best_solution = [self.bees[minimum][0], self.bees[minimum][1], michalewicz(self.bees[minimum])]

            self.evolution[0].append(it)
            self.evolution[1].append(self.best_solution[2])

    def results(self):
        print("-------------- Resultados --------------")
        print(f"x: {self.best_solution[0]} y: {self.best_solution[1]}")    
        print(f"Resultado: {self.best_solution[2]}")    
        print("----------------------------------------")  

hive = beeHive(100, 50, 0.5) # beeColony (iterations, bees, selection)
hive.run()
hive.results()

# Función
valX = np.linspace(x_min, x_max, 500)
valY = np.linspace(y_min, y_max, 500)
X, Y = np.meshgrid(valX, valY)
Z = michalewicz([X, Y])
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.scatter(hive.best_solution[0], hive.best_solution[1], hive.best_solution[2], marker='*', color='red', s=100)

# Mejores resultados
figura = plt.figure(num="ABC") 
plt.xlabel("Iteración")
plt.ylabel("Minimo encontrado")
plt.plot(hive.evolution[0], hive.evolution[1], color='red')

plt.show()