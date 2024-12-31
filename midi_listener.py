import mido 
with mido.open_input() as inport:
    for message in inport:
        print(message)

