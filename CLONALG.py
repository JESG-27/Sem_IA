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
        self.solution = []

    def run(self):
        for it in range(self.iterations):
            affinity = []
            clon_affinity = []

            # Evaluación de las soluciones
            for cell in self.cells:
                affinity.append(michalewicz(cell))
            
            # Clonación de las mejores soluciones
            best_solutions = np.argsort(affinity)[:int(len(self.cells) * self.cloneRate)]
            clones = [[self.cells[i][0], self.cells[i][1]] for i in best_solutions]
        
            # Mutación
            for clon in clones:
                for i in range(len(clon)):
                    if np.random.random() < self.mutation:
                        clon[i] += np.random.uniform(-0.5, 0.5)
                clon_affinity.append(michalewicz(clon))
            
            # Mejores soluciones mutadas
            clon_solutions = np.argsort(clon_affinity)[:len(self.cells)]
            best_clones = [[clones[i][0], clones[i][1]] for i in clon_solutions]
            for clon in best_clones:
                clones.append(clon)
            
            remain = len(self.cells)-len(clones)
            self.cells = [[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for i in range(remain)]
            for clon in clones:
                self.cells.append(clon)

        minimum = np.argmin([michalewicz([self.cells[i][0], self.cells[i][1]]) for i in range(len(self.cells))])
        self.best_solution = [self.cells[minimum][0], self.cells[minimum][1], michalewicz(self.cells[minimum])]


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

plt.show()