# Калюжный Денис Владимирович

import random


text = input("input some text: ")
words = text.split()
shuffle = []
# меняем две случайные буквы
for word in words:
    size = len(word)
    if size > 2:
        swap = [random.randrange(size), random.randrange(size)]
        if swap[0] == swap[1]:
            neword = word
        else:
            swap.sort()
            neword = f"{word[:swap[0]]}{word[swap[1]]}{word[swap[0]+1: swap[1]]}{word[swap[0]]}{word[swap[1] + 1:]}"
    else:
        neword = word
    shuffle.append(neword)
newsent = " ".join(shuffle)
print(f"shuffled: {newsent}")