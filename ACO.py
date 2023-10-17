from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Función a optimizar
def michalewicz(X):
    resultado = 0
    for i in range(len(X)):
        resultado -= np.sin(X[i]) * np.sin(((i + 1) * X[i]**2) / np.pi)**(2 * 10) # m=10
    return resultado

# Variables y valores
num_hormigas = 100
iteraciones = 500
alfa = 1.0
beta = 1.0
rho = 0.5

# Almacenamiento de resultados
feromonas = np.ones((num_hormigas, 2))
mejores_valores = None
mejor_resultado = float('inf')

# Almacenamiento para rendimiento
mejores_resultados = [], []
resultados_generales = [], []

# Algoritmo principal
for itera in range(iteraciones):
    resultados_iteracion = []
    for hormiga in range(num_hormigas):
        x = np.random.uniform(-10, 0)
        y = np.random.uniform(-10, 0)

        resultado = michalewicz([x, y])
        resultados_iteracion.append(resultado)

        if resultado < mejor_resultado:
            mejores_valores = (x, y)
            mejor_resultado = resultado
            mejores_resultados[0].append(itera)
            mejores_resultados[1].append(mejor_resultado)

        feromonas[hormiga] = (1 - rho) * feromonas[hormiga] + rho
        feromonas[hormiga] += alfa/resultado

    feromonas /= np.sum(feromonas)

    resultados_generales[0].append(itera)
    resultados_generales[1].append(resultados_iteracion[np.argmin(resultados_iteracion)])

# Graficación
figura = plt.figure(num="Michalewicz")
ejes = Axes3D(figura)
valX = np.linspace(-10, 0, 500)
valY = np.linspace(-10, 0, 500)
X, Y = np.meshgrid(valX, valY)
Z = michalewicz([X, Y])
ejes.plot_surface(X, Y, Z, cmap="viridis")
ejes.scatter(mejores_valores[0], mejores_valores[1], mejor_resultado, marker='*', color='green', s=25)

plt.figure(num="Rendimiento")
plt.xlabel("Iteración")
plt.ylabel("Minimo encontrado")
plt.plot(mejores_resultados[0], mejores_resultados[1], marker='*', color='red')
plt.scatter(resultados_generales[0], resultados_generales[1], s=10, color='blue')

# Resultados
print(f"Valores optmios: x: {mejores_valores[0]} y: {mejores_valores[1]}")
print("Mejor resultado: ", mejor_resultado)

plt.show()