from random import randint

population_size = 500

def generateRandomFella():
    fella = []
    for _ in range(20):
        fella.append(randint(0, 1))
    return fella

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

def getPopulation():
    population = []
    with open('population.txt', 'r') as file:
        for line in file:
            line = line.strip()
            ind = []
            for c in line:
                ind.append(int(c))
            population.append(ind)
    print(population, len(population))

def write():
    f = open("population.txt","w+")
    for _ in range(population_size):
        fella = generateRandomFella()
        f.write(''.join(str(v) for v in fella) + '\n')

write()
getPopulation()