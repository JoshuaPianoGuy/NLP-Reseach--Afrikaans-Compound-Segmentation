def remove_empty_lines(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.strip():  # Only write non-empty lines
                outfile.write(line)

# Usage
input_file = 'validation_set_git.txt'  # Replace with your input file path
output_file = 'val_final_git.txt'  # Replace with your desired output file path
remove_empty_lines(input_file, output_file)
