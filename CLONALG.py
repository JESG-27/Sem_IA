import numpy as np
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

class immune():
    def __init__(self, iterations, population, mutation, cloneRate):
        self.iterations = iterations
        self.cells = [[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for i in range(population)]
        self.mutation = mutation
        self.cloneRate = cloneRate
        self.evolution = [],[]
        
        minimum = np.argmin([michalewicz(self.cells[i]) for i in range(len(self.cells))])
        self.best_solution = [self.cells[minimum][0], self.cells[minimum][1], michalewicz(self.cells[minimum])]

    def run(self):
        for it in range(self.iterations):
            affinity = []

            # Evaluación de las soluciones
            for cell in self.cells:
                affinity.append(michalewicz(cell))
            
            # Clonación de las mejores soluciones
            best_solutions = np.argsort(affinity)[:int(len(self.cells) * self.cloneRate)]
            clones = [[self.cells[i][0], self.cells[i][1]] for i in best_solutions]
        
            # Mutación
            mutated_clones = clones
            for clon in mutated_clones:
                for i in range(len(clon)):
                    if np.random.random() < self.mutation:
                        clon[i] += np.random.uniform(-0.2, 0.2)

            # Nueva población
            remain = len(self.cells)-(len(clones)*2)
            self.cells = [[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for i in range(remain)]
            for clon in range(len(clones)):
                self.cells.append(clones[clon])
                self.cells.append(mutated_clones[clon])

            # Mejor solución encontrada
            minimum = np.argmin([michalewicz(self.cells[i]) for i in range(len(self.cells))])
            minimum_result = michalewicz(self.cells[minimum])
            if (minimum_result < self.best_solution[2]):
                self.best_solution = [self.cells[minimum][0], self.cells[minimum][1], minimum_result]
                        
            # Evolución
            self.evolution[0].append(it)
            self.evolution[1].append(self.best_solution[2])

clonal = immune(100, 50, 0.3, 0.1) # immune(iterations, population, mutation, cloneRate)
clonal.run()
print(clonal.best_solution)

# Función
valX = np.linspace(x_min, x_max, 500)
valY = np.linspace(y_min, y_max, 500)
X, Y = np.meshgrid(valX, valY)
Z = michalewicz([X, Y])
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.scatter(clonal.best_solution[0], clonal.best_solution[1], clonal.best_solution[2], marker='*', color='red', s=100)

# Mejores resultados
figura = plt.figure(num="DE") 
plt.xlabel("Iteración")
plt.ylabel("Minimo encontrado")
plt.plot(clonal.evolution[0], clonal.evolution[1], color='red')

plt.show()