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

class particula:
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
        self.evolucion = [],[]
        self.movimiento = [],[],[],[]

        # Creación de población inicial
        for i in range(num_particulas):
            self.particulas.append(particula())
    
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

    def pos_inicio(self):
        for part in self.particulas:
            self.movimiento[0].append(part.posicion[0])
            self.movimiento[1].append(part.posicion[1])
    
    def pos_fin(self):
        for part in self.particulas:
            self.movimiento[2].append(part.posicion[0])
            self.movimiento[3].append(part.posicion[1])

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

            self.evolucion[0].append(iteracion)    
            self.evolucion[1].append(self.mejor_resultado)

            iteracion += 1
            cont += 1
        self.iteraciones = iteracion
        
    def print_resultados(self):
        print("<------------- Resultados ------------->")
        print(f"Mejores valores: x: {self.mejores_valores[0]} y: {self.mejores_valores[1]}")
        print(f"Mejor resultado: {self.mejor_resultado}")
        print(f"Numero de iteraciones: {self.iteraciones}")
        print("<-------------------------------------->")

enjambre = PSO(500, 100)           # PSO(iteraciones, particulas)
enjambre.pos_inicio()              # Guardar posiciones iniciales
enjambre.optmizar()
enjambre.pos_fin()                 # Guardar posiciones finales
enjambre.print_resultados()

# Función
figura = plt.figure(num="Michalewicz")
ejes = Axes3D(figura)
valX = np.linspace(x_min, x_max, 500)
valY = np.linspace(y_min, y_max, 500)
X, Y = np.meshgrid(valX, valY)
Z = michalewicz([X, Y])
ejes.plot_surface(X, Y, Z, cmap="viridis")
ejes.scatter(enjambre.mejores_valores[0], enjambre.mejores_valores[1], enjambre.mejor_resultado, marker='*', color='red', s=100)

# Mejores resultados
figura = plt.figure(num="PSO") 

plt.subplot(1,2,1)
plt.plot(enjambre.evolucion[0], enjambre.evolucion[1])
plt.title("Rendimiento")
plt.xlabel("Iteración")
plt.ylabel("Mínimo")

# Movimiento
plt.subplot(1,2,2)
plt.scatter(enjambre.movimiento[0], enjambre.movimiento[1], color='red')
plt.scatter(enjambre.movimiento[2], enjambre.movimiento[3], color='green')
plt.scatter(enjambre.mejores_valores[0], enjambre.mejores_valores[1], color='blue', marker='*')
plt.title("Particulas")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()

#https://cienciadedatos.net/documentos/py02_optimizacion_pso