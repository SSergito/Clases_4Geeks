# Your code here
import math

def compute_robot_distance(moves_list):
    initial_pos = [0 , 0]
    for move in moves_list:
        if move.split()[0] == "UP":
            initial_pos[0] += int(move.split()[1])
        elif move.split()[0] == "DOWN":
            initial_pos[0] -= int(move.split()[1])
        elif move.split()[0] == "LEFT":
            initial_pos[1] -= int(move.split()[1])
        elif move.split()[0] == "RIGHT":
            initial_pos[1] += int(move.split()[1])

    movement = round(math.sqrt(abs(initial_pos[0])**2 + abs(initial_pos[1])**2))
    return movement



print(compute_robot_distance(["UP 5", "DOWN 2", "LEFT 3", "RIGHT 7"]))