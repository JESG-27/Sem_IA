import time
from algorithms import GA, ACO, PSO, DE, immune, beeHive, BFO

"""
    Arreglos para almacenar los resultados obtenidos por iteración
    algoritmo = [[Resultados], [tiempo de ejecución]]
    
    ---------------------
    Para todos los algoritmos serán los mismos parametros de ejecución:
        100 iteración
        50 soluciones candidatas
        10 corridas
    ---------------------
"""

GA_results = [[],[]]
ACO_results = [[],[]]
PSO_results = [[],[]]
DE_results = [[],[]]
CLONALG_results = [[],[]]
ABC_results = [[],[]]
BFO_results = [[],[]]

#<-------------- Genético -------------->#
for i in range(10):
    genetic = GA(100, 50)
    start = time.time()
    genetic.run()
    end = time.time()
    GA_results[0].append(genetic.best_solution[2])
    GA_results[1].append(end-start)

#<-------------- Colonia de Hormigas -------------->#
for i in range(10):
    ant_colony = ACO(100, 50, 1.0, 1.0, 0.5)
    start = time.time()
    ant_colony.run()
    end = time.time()
    ACO_results[0].append(ant_colony.best_result)
    ACO_results[1].append(end-start)

# #<-------------- Enjambre de Particulas -------------->#
for i in range(10):
    particules = PSO(100, 50)
    start = time.time()
    particules.optmizar()
    end = time.time()
    PSO_results[0].append(particules.mejor_resultado)
    PSO_results[1].append(end-start)
    
# #<-------------- Evolución Diferencial -------------->#
for i in range(10):
    diferential = DE(50, 100)
    start = time.time()
    diferential.run()
    end = time.time()
    DE_results[0].append(diferential.best_global_solution[2])
    DE_results[1].append(end-start)


# #<-------------- Selección Clonal -------------->#
for i in range(10):
    clonal = immune(100, 50, 0.3, 0.15)
    start = time.time()
    clonal.run()
    end = time.time()
    CLONALG_results[0].append(clonal.best_solution[2])
    CLONALG_results[1].append(end-start)

# #<-------------- Colonia de Abejas -------------->#
for i in range(10):
    hive = beeHive(100, 50, 0.3)
    start = time.time()
    hive.run()
    end = time.time()
    ABC_results[0].append(hive.best_solution[2])
    ABC_results[1].append(end-start)

# #<-------------- Forrajeo de Bacterias -------------->#
for i in range(10):
    bacteria = BFO(50, 100, 0.3)
    start = time.time()
    bacteria.run()
    end = time.time()
    BFO_results[0].append(bacteria.best_global_solution[2])
    BFO_results[1].append(end-start)