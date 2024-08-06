separator = ","
file_name = "maze_small.txt"
output_file_name = file_name.replace("txt", "csv")

with open(output_file_name, "w") as output_file:
    with open(file_name) as maze_file:
        for line in maze_file:
            if line[0] == '#':
                continue
            output_file.write(separator.join(list(line)))

