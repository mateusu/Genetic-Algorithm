import random
import math
import numpy as np
import time
from random import randint
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


## OPÇÕES DE POPULAÇÃO ##

# tamanho da população inicial
population_size = 100

#quantas vezes irá rodar o algorítmo
repeat = 10

# máximo de gerações
max_generations = 200


## OPÇÕES DE ALGORÍTMOS DE SELEÇÃO ##

# número de indivíduos escolhidos na seleção por torneio #
battle_royale_select = int(population_size/10)

# tipo de algorítmos de seleção
# 0 = roullete
# 1 = torneio
selection_type = 0


## OPÇÕES DE CROSSOVER ##

# máscara de onde acontecerá o crossover
crossover_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# chance de o crossover acontecer
crossover_chance = 0.7


## OPÇÕES DE MUTAÇÃO ##

# 0 = muta n genes aleatórios
# 1 = percorre cada gene, decide se vai mutar ou não
mutation_type = 0

# número de genes a serem mutados
mutation_number = 2

# chance de acontecer a mutação
mutation_chance = 0.05


## FUNÇÕES UTILITÁRIAS ##

# Generate Random Fella: gera um indivíduo aleatório

def generateRandomFella():
    fella = []
    for _ in range(20):
        fella.append(randint(0, 1))
    return fella

# Choose Selection Algorithm: seleciona o tipo de algorítmo escolhido


def chooseSelectionAlgorithm(population, population_chance, population_results):
    fella = 0

    if selection_type == 0:
        fella = selectFellaRoullete(population_chance)

    if selection_type == 1:
        fella = selectFellaBR(population, population_results)

    return fella


# Get Decimal Value: retorna o valor decimal de X e Y do cromossomo

def getDecimalValue(ind):
    xBits = ind[:len(ind)//2]
    yBits = ind[len(ind)//2:]
   
    x = convertValue(xBits)
    y = convertValue(yBits)
    return x, y

def convertValue(bitValue):
    decValue = int("".join(str(x) for x in bitValue), 2)
    decValue = float(decValue)
    decValue = ((decValue) * 0.00978) - 5
    return decValue
    
# Get Best Generation Fella: retorna o melhor indivíduo da geração, e seus valores


def getBestGenerationFella(population, population_results, population_fitness):
    best_result = min(population_results)
    best_fitness = population_fitness[population_results.index(best_result)]
    best_fella = population[population_results.index(best_result)]
    return best_result, best_fella, best_fitness


# Print Retults: printa os resultados

def printResults(best_result, best_fella, best_fitness):

    x, y = getDecimalValue(best_fella)
    print('Melhor resultado:')
    print('X = ', x, 'Y =', y, 'F(X,Y) =', best_result)
    print('Melhor fitness =', best_fitness)
    print('')


## FUNÇÕES DE FITNESS ##

# Get Population Fitness: calcula o fitness de toda a população da geração atual
# results = resultado da função de rastrigin
# fitness = fitness do indivíduo
# chance = chance em porcentagem da escolha do indivíduo

def getPopulationFitness(population):
    population_results = []
    population_fitness = []
    population_chance = []

    for ind in population:
        result, fitness = getFitness(ind)
        population_results.append(result)
        population_fitness.append(fitness)

    for ind_fitness in population_fitness:
        population_chance.append((ind_fitness / sum(population_fitness)))

    return population_chance, population_results, population_fitness

# Get Fitness: retorna o fitness do invidíduo, além do seu resultado em rastrigin


def getFitness(ind):
    xValue, yValue = getDecimalValue(ind)
    result = rastrigin(xValue, yValue)

    fitness = 100 - result
    
    return result, fitness

# Rastrigin: função de rastrigin


def rastrigin(x, y):
    return 20 + (x ** 2) + (y ** 2) - (10 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))


## ALGORÍTMOS DE SELEÇÃO ##

# Roleta: escolhe um indivíduo aleatóriamente, considerando os pesos

def selectFellaRoullete(population_chance):
    roullete_list = []
    sum = 0
    for prob in population_chance:
        sum += prob
        roullete_list.append(sum)

    fella = roullete_spin(roullete_list)

    return fella

# Roleta: spin da roleta

def roullete_spin(roullete_list):
    sorted_number = random.uniform(0, 1)
    sorted_guy = 0

    for i in range(len(roullete_list)):
        if sorted_number <= roullete_list[i]:
            sorted_guy = i
            break
        elif i == (len(roullete_list) - 1) and sorted_number <= 1:
            sorted_guy = i

    return sorted_guy

# Torneio: escolhe 20% da população aleatóriamente, seleciona o melhor dentre eles

def selectFellaBR(population, population_results):
    fellas = []
    selected = []
    selected_fitness = []
    for i in range(battle_royale_select):
        fellas.append(randint(0, population_size-1))

    for i in range(0, len(fellas)):
        selected.append(population[fellas[i]])
        selected_fitness.append(population_results[fellas[i]])
    
    min_fitness = min(selected_fitness)
    fella = selected[selected_fitness.index(min_fitness)]
    fella = population.index(fella)

    return fella


## FUNÇÕES DE MUTAÇÃO ##

# Random Mutation: decide de haverá mutação, se sim, muta n genes aleatórios
def randomMutation(children):
    chance = random.uniform(0, 1)
    if chance <= mutation_chance:
        genes = []
        for _ in range(mutation_number):

            value = randint(0, len(children)-1)
            while value in genes:
                value = randint(0, len(children)-1)
                if value not in genes:
                    break

            genes.append(value)

        for j in genes:
            if children[j] == 1:
                children[j] = 0
            else:
                children[j] = 1

    return children

# Iterated Mutation: percorre cada gene e decide se mutará ou não


def iteratedMutation(children):
    for i in range(len(children)):

        chance = random.uniform(0, 1)

        if chance <= mutation_chance:
            if children[i] == 0:
                children[i] = 1
            else:
                children[i] = 0

    return children


## FUNÇÃO DE CROSSOVER ##

# Crossover: faz crossover (ou não) de 2 indivíduos de acordo com a crossover_mask

def crossover(chromo1, chromo2):
    chance = random.uniform(0, 1)

    if chance <= crossover_chance:
        child1 = []
        child2 = []

        for i in range(len(crossover_mask)):
            if crossover_mask[i] == 1:
                child2.append(chromo1[i])
                child1.append(chromo2[i])
            else:
                child1.append(chromo1[i])
                child2.append(chromo2[i])

        if mutation_type == 0:
            child1 = randomMutation(child1)
            child2 = randomMutation(child2)

        if mutation_type == 1:
            child1 = iteratedMutation(child1)
            child2 = iteratedMutation(child2)

        return child1, child2

    else:

        return chromo1, chromo2


## FUNÇÕES PRINCIPAIS ##

# Start New Generation: Gera uma nova geração, utilizando algum dos algorítmos de seleção

def startNewGeneration(population, population_chance, population_results):

    new_population = []

    while len(new_population) < len(population):

        x = chooseSelectionAlgorithm(
            population, population_chance, population_results)
        y = chooseSelectionAlgorithm(
            population, population_chance, population_results)

        x, y = crossover(population[x], population[y])

        new_population.append(x)
        new_population.append(y)
        if len(new_population) == len(population):
            break

    return new_population


def start():

    start = time.time()

    population_init = []
    best_fitness_list = []
    average_fitness_list = []

    for _ in range(population_size):
        fella = generateRandomFella()
        population_init.append(fella)

    for i in range(repeat):
        population = population_init
        generation = 1
        best_fitness_per_generation = []
        average_fitness_per_generation = []

        for _ in range(max_generations):
            
            population_chance, population_results, population_fitness = getPopulationFitness(
                population)

            best_result, best_fella, best_fitness = getBestGenerationFella(
                population, population_results, population_fitness)

            print('Geração:', generation)
            printResults(best_result, best_fella, best_fitness)

            best_fitness_per_generation.append(best_fitness)
            average_fitness = sum(population_fitness)/len(population_fitness)
            average_fitness_per_generation.append(average_fitness)

            population = startNewGeneration(
                population, population_chance, population_results)

            generation += 1

        best_fitness_list.append(best_fitness_per_generation)
        average_fitness_list.append(average_fitness_per_generation)


    end = time.time()

    cleanData(best_fitness_list, average_fitness_list)
  
    print('Tempo de execução:', end - start)


def cleanData(best_fitness_list, average_fitness_list):
    
    average_fitness = []
    best_fitness = []
    
    for i in range(repeat):
        bt_vector = np.array(best_fitness_list[i])
        av_vector = np.array(average_fitness_list[i])
        best_fitness.append(bt_vector)
        average_fitness.append(av_vector)
    
    best_fitness = sum(best_fitness)
    average_fitness = sum(average_fitness)

    best_fitness = best_fitness/repeat
    average_fitness = average_fitness/repeat

    plotGraph(best_fitness, average_fitness)


def plotGraph(best_fitness, average_fitness):
    generations = [i+1 for i in range(max_generations)]

    plt.plot(generations, average_fitness,
             color='orange', label='Average')
    plt.plot(generations, best_fitness,
             color='blue', label='Best')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness per generation')
    plt.legend(loc='lower right')

    selectiontype = "roullete" if selection_type == 0 else "tournament"
    mutationtype = "n bits mutation" if mutation_type == 0 else "bit to bit"
    mutatedgenes = mutation_number if mutation_type == 0 else "random"
    run_config = AnchoredText("population_size={}\n\nselection_type={}\nmutation_type={}\n\nmutation_chance={}\nmutated_genes={}\ncrossover_chance={}".format(population_size, selectiontype, mutationtype, mutation_chance, mutatedgenes, crossover_chance), 
                loc=5, pad=0.4, 
                borderpad=2)

    plt.gca().add_artist(run_config)

    plt.show()

start()
