def lyrics_generator(lista):
    rhythm = []
    bass_counter = 0
    for i in lista:
        if i == 1:
            rhythm.append("Drop the bass")
            bass_counter += 1
        elif i == 0:
            rhythm.append("Boom")
            bass_counter = 0
        if bass_counter == 3:
            rhythm.append("!!!Break the bass!!!")
            bass_counter = 0
    return " ".join(rhythm)

# Your code above, nothing to change after this line
print(lyrics_generator([0,0,1,1,0,0,0]))
print(lyrics_generator([0,0,1,1,1,0,0,0]))
print(lyrics_generator([0,0,0]))
print(lyrics_generator([1,0,1]))
print(lyrics_generator([1,1,1]))
