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

class BFO:
    def __init__(self, bacteria, iterations, chemotaxis):
        self.bacteria = [[np.random.uniform(x_min, x_max), random.uniform(y_min, y_max)] for i in range(bacteria)]
        self.iterations = iterations
        self.chemotaxis = chemotaxis
        self.movement = [],[],[],[]
        
        minimum = np.argmin([michalewicz([self.bacteria[i][0], self.bacteria[i][1]]) for i in range(len(self.bacteria))])
        self.best_global_solution = [self.bacteria[minimum][0], self.bacteria[minimum][1], michalewicz(self.bacteria[minimum])]
    
    def borders(self, bacterum):
        if (bacterum[0] > x_max):
            bacterum[0] = x_max
        elif (bacterum[0] < x_min):
            bacterum[0] = x_min

        if (bacterum[1] > y_max):
            bacterum[0] = y_max
        elif (bacterum[1] < y_min):
            bacterum[1] = y_min
    
    def start_pos(self):
        for bacterum in self.bacteria:
            self.movement[0].append(bacterum[0])
            self.movement[1].append(bacterum[1])
    
    def end_pos(self):
        for bacterum in self.bacteria:
            self.movement[2].append(bacterum[0])
            self.movement[3].append(bacterum[1])
    
    def results(self):
        print("<------------- Resultados ------------->")
        print(f"Mejores valores: x: {self.best_global_solution[0]} y: {self.best_global_solution[1]}")
        print(f"Mejor resultado: {self.best_global_solution[2]}")
        print("<-------------------------------------->")
    
    def run(self):
        self.start_pos()
        for iteration in range(self.iterations):
            for bacterum in self.bacteria:
            # Quimiotaxis
                minimum = np.argmin([michalewicz([self.bacteria[i][0], self.bacteria[i][1]]) for i in range(len(self.bacteria))])
                best_bacteria = self.bacteria[minimum]
                bacterum[0] += self.chemotaxis * (best_bacteria[0] - bacterum[0])
                bacterum[1] += self.chemotaxis * (best_bacteria[1] - bacterum[1])
        
            # Reproducción
                # Nueva bacteria
                aux = self.bacteria.copy()
                aux.pop(minimum)
                second_minimum = np.argmin([michalewicz([aux[i][0], aux[i][1]]) for i in range(len(aux))])
                
                if (second_minimum > minimum):
                    second_minimum += 1
                
                new_bacteria = [(self.bacteria[minimum][0] + self.bacteria[second_minimum][0])/2, (self.bacteria[minimum][1] + self.bacteria[second_minimum][1])/2]
                self.bacteria.append(new_bacteria)
                
                # Eliminar bacteria
                maximum = np.argmax([michalewicz([self.bacteria[i][0], self.bacteria[i][1]]) for i in range(len(self.bacteria))])
                self.bacteria.pop(maximum)
                
            # Mutación
            for bacterum in self.bacteria:
                for i in range(len(bacterum)):
                    if random.random() < self.chemotaxis:
                        bacterum[i] += np.random.uniform(-1,1)
                
                # Limites de busqueda
                self.borders(bacterum)

        minimum = np.argmin([michalewicz([self.bacteria[i][0], self.bacteria[i][1]]) for i in range(len(self.bacteria))])
        self.best_global_solution = [self.bacteria[minimum][0], self.bacteria[minimum][1], michalewicz(self.bacteria[minimum])]
        self.end_pos()
        
optimization = BFO(50, 100, 0.5) # BFO(bacterias, iteraciones)
optimization.run()
optimization.results()

# Función
valX = np.linspace(x_min, x_max, 500)
valY = np.linspace(y_min, y_max, 500)
X, Y = np.meshgrid(valX, valY)
Z = michalewicz([X, Y])
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.scatter(optimization.best_global_solution[0], optimization.best_global_solution[1], optimization.best_global_solution[2], marker='*', color='red', s=100)

fig = plt.figure()
plt.scatter(optimization.movement[0], optimization.movement[1], color='red')
plt.scatter(optimization.movement[2], optimization.movement[3], color='green')
plt.xlabel("X")
plt.ylabel("Y")

plt.show()