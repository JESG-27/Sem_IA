from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def esfera():
    figura = plt.figure(num="Esfera")
    ejes = Axes3D(figura)

    valX = np.linspace(-10, 10, 500)
    valY = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(valX, valY)
    

    ejes.plot_surface(X, Y, ((X**2)+(Y**2)), cmap="viridis")

def step():
    figura = plt.figure(num="Step")
    ejes = Axes3D(figura)

    valX = np.linspace(-100, 100, 500)
    valY = np.linspace(-100, 100, 500)
    X, Y = np.meshgrid(valX, valY)

    ejes.plot_surface(X, Y, ((np.floor(X+0.5))**2+(np.floor(Y+0.5))**2), cmap="viridis")


def absolute():
    figura = plt.figure(num="Absolute")
    ejes = Axes3D(figura)

    valX = np.linspace(-100, 100, 500)
    valY = np.linspace(-100, 100, 500)
    X, Y = np.meshgrid(valX, valY)

    ejes.plot_surface(X, Y, (np.absolute(X) + (np.absolute(Y))), cmap="viridis")

def michalewicz():
    def calcular_Z(X):
        resultado = 0
        for i in range(len(X)):
            resultado -= np.sin(X[i]) * np.sin(((i + 1) * X[i]**2) / np.pi)**(2 * 10) # m=10
        return resultado
    
    figura = plt.figure(num="Michalewicz")
    ejes = Axes3D(figura)

    valX = np.linspace(0, np.pi, 1500)
    valY = np.linspace(0, np.pi, 1500)
    X, Y = np.meshgrid(valX, valY)
    Z = calcular_Z([X, Y])

    ejes.plot_surface(X, Y, Z, cmap="viridis")

def eggholder():
    figura = plt.figure(num="Eggholder")
    ejes = Axes3D(figura)

    valX = np.linspace(-500, 500, 500)
    valY = np.linspace(-500, 500, 500)
    X, Y = np.meshgrid(valX, valY)
    Z = -(Y+47) * np.sin(np.sqrt(np.absolute((X/2)+(Y+47)))) - X * np.sin(np.sqrt(np.absolute(X-(Y+47))))

    ejes.plot_surface(X, Y, Z, cmap="viridis")

def weierstrass():
    def calcular_Z(x, y):
        resultado = 0
        a = 0.5
        b = 3
        kMax = 20
        for i in range(kMax):
            resultado += a**i * np.cos(b**i * np.pi * x) * np.cos(b**i * np.pi * y)
        return resultado


    figura = plt.figure(num="Weierstrass")
    ejes = Axes3D(figura)

    valX = np.linspace(-5, 5, 100)
    valY = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(valX, valY)
    Z = np.vectorize(calcular_Z)(X, Y)

    ejes.plot_surface(X, Y, Z, cmap="viridis")

esfera()
step()
absolute()
michalewicz()
eggholder()
weierstrass()

plt.show()