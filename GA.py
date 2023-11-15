import numpy as np
import random
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

class GE:
    def __init__(self, iterations, genes):
        self.iterations = iterations
        self.genes = [[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for i in range(genes)]
        self.fitness = [michalewicz(self.genes[i]) for i in range(len(self.genes))]
        minimum = np.argmin(self.fitness)
        self.best_solution = [self.genes[minimum][0], self.genes[minimum][1], self.fitness[minimum]]
        self.evolution = [[],[]]

    def run(self):
        for it in range(self.iterations):
            # Evaluación
            for i in range(len(self.genes)):
                self.fitness[i] = michalewicz(self.genes[i])

            # Selección
            index = np.argsort(self.fitness)[:len(self.genes)//3]
            best_genes = [[self.genes[i][0], self.genes[i][1]] for i in index]
            
            # Recombinación
            recombination = []
            for i in range(len(best_genes)//2):
                a, b = best_genes[2*i], best_genes[2*i+1]
                recombination_point = random.randint(0, len(a)-1)
                recombination.append(a[:recombination_point] + b[recombination_point:])
                recombination.append(b[:recombination_point] + a[recombination_point:])
            
            # Mutacion
            mutated = recombination
            for gen in mutated:
                for i in range(len(gen)):
                    if random.random() < 0.5:
                        gen[i] += np.random.uniform(-1,1)
            
            # Nueva población
            self.genes = best_genes + recombination + mutated

            minimum = np.argmin(self.fitness)
            best_solution = self.fitness[minimum]
            self.evolution[0].append(it)
            self.evolution[1].append(best_solution)
            
        minimum = np.argmin(self.fitness)
        self.best_solution = [self.genes[minimum][0], self.genes[minimum][1], self.fitness[minimum]]

optimization = GE(100, 50)
optimization.run()

# Función
valX = np.linspace(x_min, x_max, 500)
valY = np.linspace(y_min, y_max, 500)
X, Y = np.meshgrid(valX, valY)
Z = michalewicz([X, Y])
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.scatter(optimization.best_solution[0], optimization.best_solution[1], optimization.best_solution[2], marker='*', color='red', s=100)

# Mejores resultados
figura = plt.figure(num="DE") 
plt.xlabel("Iteración")
plt.ylabel("Minimo encontrado")
plt.plot(optimization.evolution[0], optimization.evolution[1], color='red')


plt.show()