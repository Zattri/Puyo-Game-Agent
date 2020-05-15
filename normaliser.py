from random import shuffle

def normaliseActionsFromFile(data):

    a_inputs = []
    b_inputs = []
    left_inputs = []
    right_inputs = []
    down_inputs = []
    other_inputs = []

    # Split data into different input actions
    for item in data:
        action_number = item[1][3]
        if action_number == 0:
            b_inputs.append(item)
        elif action_number == 1:
            a_inputs.append(item)
        elif action_number == 2:
            down_inputs.append(item)
        elif action_number == 3:
            left_inputs.append(item)
        elif action_number == 4:
            right_inputs.append(item)
        else:
            other_inputs.append(item)

    # print(f"A: {len(a_inputs)} | B: {len(b_inputs)} | LEFT: {len(left_inputs)} | RIGHT: {len(right_inputs)} | DOWN: {len(down_inputs)} | OTHER: {len(other_inputs)}")

    #Find the smallest length
    smallest_array_count = len(a_inputs)

    if (len(b_inputs) < smallest_array_count):
        smallest_array_count = len(b_inputs)

    if (len(left_inputs) < smallest_array_count):
        smallest_array_count = len(left_inputs)

    if (len(right_inputs) < smallest_array_count):
        smallest_array_count = len(right_inputs)


    # Shuffle all arrays before sampling
    shuffle(a_inputs)
    shuffle(b_inputs)
    shuffle(left_inputs)
    shuffle(right_inputs)


    # Reduce array count to lowest
    reduced_a = a_inputs[:smallest_array_count]
    reduced_b = b_inputs[:smallest_array_count]
    reduced_left = left_inputs[:smallest_array_count]
    reduced_right = right_inputs[:smallest_array_count]

    # print(f"SHUFFLED LENS - A: {len(reduced_a)} | B: {len(reduced_b)} | LEFT: {len(reduced_left)} | RIGHT: {len(reduced_right)} | DOWN: {len(down_inputs)} | OTHER: {len(other_inputs)}")

    combined_data = reduced_a[:] + reduced_b[:] + reduced_left[:] + reduced_right[:]

    shuffle(combined_data)
    return combined_data
