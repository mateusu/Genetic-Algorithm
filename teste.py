import random


def roullete(probs):
    roullete_list = []
    sum = 0
    for prob in probs:
        sum += prob
        roullete_list.append(sum)

    roullete_spin(roullete_list)


def roullete_spin(roullete_list):
    sorted_number = 0.98
    sorted_guy = 0

    for i in range(len(roullete_list)):
        if sorted_number <= roullete_list[i]:
            sorted_guy = i
            break
        elif i == (len(roullete_list) - 1) and sorted_number <= 1:
            sorted_guy = i

    return sorted_guy
roullete([0.2, 0.1, 0.1, 0.5, 0.05])
