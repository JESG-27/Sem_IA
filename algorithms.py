import numpy as np
import random

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

class GA:
    def __init__(self, iterations, genes):
        self.iterations = iterations
        self.genes = [[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for i in range(genes)]
        self.fitness = [michalewicz(self.genes[i]) for i in range(len(self.genes))]
        minimum = np.argmin(self.fitness)
        self.best_solution = [self.genes[minimum][0], self.genes[minimum][1], self.fitness[minimum]]

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
            remain = len(self.genes) - (len(best_genes)+len(recombination)+len(mutated))
            self.genes = best_genes + recombination + mutated
            for i in range(remain):
                self.genes.append([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])  

            minimum = np.argmin(self.fitness)
            best_solution = self.fitness[minimum]
            
        minimum = np.argmin(self.fitness)
        self.best_solution = [self.genes[minimum][0], self.genes[minimum][1], self.fitness[minimum]]
        
class ACO:
    def __init__(self, iterations, ants, alfa, beta, rho):
        self.iterations = iterations
        self.ants = [[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for i in range(ants)]
        self.alfa = alfa
        self.beta = beta
        self.rho = rho
        self.pheromones = np.ones((ants, 2))
    
    def run(self):
        self.best_solution = None
        self.best_result = float('inf')

        # Algoritmo principal
        for itera in range(self.iterations):
            for ant in self.ants:
                x = np.random.uniform(-10, 0)
                y = np.random.uniform(-10, 0)

                result = michalewicz([x, y])

                if result < self.best_result:
                    best_solution = (x, y)
                    self.best_result = result
    

class particule:
    def __init__(self):
        self.posicion = [np.random.uniform(x_max,x_min), np.random.uniform(y_max, y_min)]
        self.velocidad_x = 0
        self.velocidad_y = 0
        self.resultado = michalewicz([self.posicion[0], self.posicion[1]])
        self.mejor = [self.resultado, self.posicion[0], self.posicion[1]]

class PSO:
    def __init__(self, iteraciones, num_particulas):
        self.iteraciones = iteraciones
        self.num_particulas = num_particulas
        self.particulas = []
        self.mejor_resultado = float('inf')
        self.mejores_valores = []

        # Creación de población inicial
        for i in range(num_particulas):
            self.particulas.append(particule())
    
    def evaluar(self, part):
        part.resultado = michalewicz([part.posicion[0],part.posicion[1]])
        if ( part.resultado < part.mejor[0] ):
            part.mejor[0] = part.resultado
            part.mejor[1] = part.posicion[0]
            part.mejor[2] = part.posicion[1]
    
    def limites(self, part):
        velocidad_max_x = (x_max-x_min)*0.2
        velocidad_min_x = -velocidad_max_x
        velocidad_max_y = (y_max-y_min)*0.2
        velocidad_min_y = -velocidad_max_y

        if (part.posicion[0] > x_max):
            part.posicion[0] = x_max
        elif (part.posicion[0] < x_min):
            part.posicion[0] = x_min

        if (part.posicion[1] > y_max):
            part.posicion[0] = y_max
        elif (part.posicion[1] < y_min):
            part.posicion[1] = y_min
        
        if (part.velocidad_x > velocidad_max_x or part.velocidad_x < velocidad_min_x):
            part.velocidad_x = 0

        if (part.velocidad_y > velocidad_max_y or part.velocidad_y < velocidad_min_y):
            part.velocidad_y = 0

    def optmizar(self):
        cont = 0
        iteracion = 0
        inercia_max = 1.2
        inercia_min = 0.2
        cognitivo = 2
        r1_x = np.random.uniform(0, 1)
        r1_y = np.random.uniform(0, 1)
        social = 2
        r2_x = np.random.uniform(0, 1)
        r2_y = np.random.uniform(0, 1)
        
        while iteracion < self.iteraciones and cont < self.iteraciones/2:
            for part in self.particulas:
                self.evaluar(part)
                if part.resultado < self.mejor_resultado:
                    self.mejor_resultado = part.resultado
                    self.mejores_valores = part.posicion
                    cont = 0
                
                inercia = inercia_max - iteracion * ((inercia_max-inercia_min)/self.iteraciones)

                nueva_velocidad_x = (
                    inercia * part.velocidad_x 
                    + cognitivo * r1_x * (part.mejor[1] - part.posicion[0]) 
                    + social * r2_x * (self.mejores_valores[0] - part.posicion[0])
                    )
                
                nueva_velocidad_y = (
                    inercia * part.velocidad_y
                    + cognitivo * r1_y * (part.mejor[2] - part.posicion[1]) 
                    + social * r2_y * (self.mejores_valores[1] - part.posicion[1])
                    )
                
                part.posicion[0] += nueva_velocidad_x
                part.posicion[1] += nueva_velocidad_y

                self.limites(part)

            iteracion += 1
            cont += 1
        self.iteraciones = iteracion        

class DE:
    def __init__(self, individuals, iterations):
        self.individuals = individuals
        self.iterations = iterations
        self.crossover = 0.2
        self.population = np.random.uniform(low=[x_min, y_min], high=[x_max , y_max], size=(self.individuals, 2))
        
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
    
    def borders(self, ind):
        if (x_min <= ind[0] and ind[0] <= x_max and y_min <= ind[1] and ind[1] <= y_max):
            return True
        else:
            return False


class immune():
    def __init__(self, iterations, population, mutation, cloneRate):
        self.iterations = iterations
        self.cells = [[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for i in range(population)]
        self.mutation = mutation
        self.cloneRate = cloneRate
        
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


class beeHive:
    def __init__(self, iterations, bees, selection):
        self.iterations = iterations
        self.bees = [[np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)] for i in range(bees)]
        self.bees_cont = [0 for i in range(bees)]
        self.selection = selection
        self.limit = bees//4
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
                if random.random() < self.selection:
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



class BFO:
    def __init__(self, bacteria, iterations, chemotaxis):
        self.bacteria = [[np.random.uniform(x_min, x_max), random.uniform(y_min, y_max)] for i in range(bacteria)]
        self.iterations = iterations
        self.chemotaxis = chemotaxis
        
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
    
    def run(self):
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