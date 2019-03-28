import random
import math
from random import randint
import matplotlib.pyplot as plt


## OPÇÕES DE POPULAÇÃO ##

# tamanho da população inicial
population_size = 20

# máximo de gerações
max_generations = 200


## OPÇÕES DE ALGORÍTMOS DE SELEÇÃO ##

# número de indivíduos escolhidos na seleção por torneio #
battle_royale_select = int(population_size/4)

# tipo de algorítmos de seleção
# 0 = roullete
# 1 = torneio
selection_type = 1


## OPÇÕES DE CROSSOVER ##

# máscara de onde acontecerá o crossover
crossover_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# chance de o crossover acontecer
crossover_chance = 0.7


## OPÇÕES DE MUTAÇÃO ##

# 0 = muta n genes aleatórios
# 1 = percorre cada gene, decide se vai mutar ou não
mutation_type = 1

# número de genes a serem mutados
mutation_number = 3

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
    xValue = int("".join(str(x) for x in xBits), 2)
    xValue = float(xValue)
    yValue = int("".join(str(x) for x in yBits), 2)
    yValue = float(yValue)
    xValue = (xValue - 512) / 100
    yValue = (yValue - 512) / 100
    return xValue, yValue

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

    fitness = 0

    if result > 0:
        fitness = (1/result)
    else:
        fitness = (1 / 0.01)

    return result, fitness

# Rastrigin: função de rastrigin

def rastrigin(x, y):
    return 20 + (x ** 2) + (y ** 2) - (10 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))


## ALGORÍTMOS DE SELEÇÃO ##

# Roleta: escolhe um indivíduo aleatóriamente, considerando os pesos

def selectFellaRoullete(population_chance):
    roullete = []

    for i in range(len(population_chance)):
        times = int(population_chance[i] * 1000)
        for _ in range(times):
            roullete.append(i)

    selected = randint(0, len(roullete)-1)
    fella = roullete[selected]

    return fella

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
    
    population = []
    generation = 1
    generations = []
    best_fitness_per_generation = []
    average_fitness_per_generation = []

    for _ in range(population_size):
        population.append(generateRandomFella())

    for _ in range(max_generations):
        population_chance, population_results, population_fitness = getPopulationFitness(
            population)

        best_result, best_fella, best_fitness = getBestGenerationFella(
            population, population_results, population_fitness)

        print('Geração:', generation)
        printResults(best_result, best_fella, best_fitness)

        generations.append(generation)
        best_fitness_per_generation.append(best_fitness)
        average_fitness = sum(population_fitness)/len(population_fitness)
        average_fitness_per_generation.append(average_fitness)

        population = startNewGeneration(
            population, population_chance, population_results)

        generation += 1
   
    plt.plot(generations, average_fitness_per_generation, color='g', label='Average')
    plt.plot(generations, best_fitness_per_generation, color='orange', label='Best')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness per generation')
    plt.legend(loc='lower right')
    plt.show()

start()
