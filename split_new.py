import re
import random
from collections import defaultdict

def extract_root_word(word):
    # Remove any spaces around morpheme boundaries and compound boundaries
    word = re.sub(r"\s*_\s*", "_", word)
    word = re.sub(r"\s*\+\s*", "+", word)
    
    # If there's a morpheme boundary, take the part before the first underscore
    if "_" in word:
        return word.split("_")[0].strip()
    
    # If there's no morpheme boundary but a plus sign, take the part before the first plus sign
    if "+" in word:
        return word.split("+")[0].strip()
    
    # If neither morpheme boundary nor plus sign exists, treat the whole word as root
    return word.strip()

def process_word(word):
    # Remove all spaces, morpheme boundaries, and hyphens
    word = re.sub(r"\s+", "", word)  # Remove all spaces
    word = re.sub(r"_", "", word)  # Remove morpheme boundaries
    word = word.replace("-", "")  # Remove hyphens
    return word

def create_word_pair(word):
    # Create the word without any annotations (including + signs)
    word_no_annotations = process_word(word.replace("+", ""))
    
    # Create the word with @ signs replacing + signs
    word_with_at = process_word(word.replace("+", "@"))
    
    return f"{word_no_annotations}\t{word_with_at}"

def group_words_by_root(input_file):
    groups = defaultdict(list)

    # Read the input file and group by root word
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                root_word = extract_root_word(word)
                word_pair = create_word_pair(word)
                groups[root_word].append(word_pair)
    
    return groups

def distribute_groups(groups, train_ratio=0.8, val_ratio=0.1):
    # Shuffle the root word groups
    grouped_items = list(groups.items())
    random.shuffle(grouped_items)

    # Calculate split sizes
    total_groups = len(grouped_items)
    train_size = int(total_groups * train_ratio)
    val_size = int(total_groups * val_ratio)
    
    # Split into train, validation, and test sets
    train_set = grouped_items[:train_size]
    val_set = grouped_items[train_size:train_size + val_size]
    test_set = grouped_items[train_size + val_size:]

    return train_set, val_set, test_set

def save_sets_to_files(train_set, val_set, test_set, train_file, val_file, test_file):
    def write_set(data_set, file_name):
        with open(file_name, 'w', encoding='utf-8') as f_out:
            for root_word, word_pairs in data_set:
                for word_pair in word_pairs:
                    f_out.write(f"{word_pair}\n")
                f_out.write("\n")
    
    # Save each set to its respective file
    write_set(train_set, train_file)
    write_set(val_set, val_file)
    write_set(test_set, test_file)

if __name__ == "__main__":
    input_file = "Dataset No Duplicates.txt"  # Replace with your actual input file

    # Group words by root word
    grouped_words = group_words_by_root(input_file)

    # Randomly distribute groups into train, validation, and test sets
    train_set, val_set, test_set = distribute_groups(grouped_words)

    # Save the sets to their respective files
    save_sets_to_files(train_set, val_set, test_set, 
                       "train_set_git.txt", "validation_set_git.txt", "test_set_git.txt")

    print("Words have been grouped, processed, and distributed into train, validation, and test sets.")