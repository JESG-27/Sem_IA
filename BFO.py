import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
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

class BFOA:
    def __init__(self, bacteria, iterations, chemotaxis):
        self.bacteria = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(bacteria, 2))
        self.iterations = iterations
        self.chemotaxis = chemotaxis
        
        minimun = np.argmin([michalewicz(self.bacteria[i]) for i in range(len(self.bacteria))])
        self.best_global_solution = [self.bacteria[minimun][0], self.bacteria[minimun][1], michalewicz(self.bacteria[minimun])]
    
    def run(self):
        for iteration in range(self.iterations):
            for bacterum in self.bacteria:
                # Quimiotaxis
                minimun = np.argmin([michalewicz(self.bacteria[i]) for i in range(len(self.bacteria))])
                best_bacteria = self.bacteria[minimun]
                bacterum += self.chemotaxis * (best_bacteria - bacterum)
        
            # Reproducción
            sample1, sample2 = random.sample(range(len(self.bacteria)), 2)
            new_bacteria = (self.bacteria[sample1] + self.bacteria[sample2]) / 2
            self.bacteria = np.vstack([self.bacteria, new_bacteria])
            
            # Mutación
            for bacterum in self.bacteria:
                if random.random() < self.chemotaxis:
                    bacterum += np.random.uniform(-1,1)
            
            minimun = np.argmin([michalewicz(self.bacteria[i]) for i in range(len(self.bacteria))])
            self.best_global_solution = [self.bacteria[minimun][0], self.bacteria[minimun][1], michalewicz(self.bacteria[minimun])]

            # if (best_solution[2] < self.best_global_solution[2]):
            #     self.best_global_solution = best_solution


optimization = BFOA(10, 50, 0.5)
optimization.run()
print(len(optimization.bacteria))
print(optimization.best_global_solution)

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