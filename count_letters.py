# --- Goal
# Write a Python program that prints the relative frequence of each letter
# of the alphabet (without distinguishing between lower and upper case) in the
# book.

# --- Specifications
# - the program should have a --help option summarizing the usage
# - the program should accept the path to the input file from the command line
# - the program should print out the total elapsed time
# - the program should have an option to display a histogram of the frequences
# - [optional] the program should have an option to skip the parts of the text
#   that do not pertain to the book (e.g., preamble and license)
# - [optional] the program should have an option to print out the basic book
#   stats (e.g., number of characters, number of words, number of lines, etc.)

import argparse
import string
import time

def skip_preamble(text):

    start_marker = "*** START OF THE PROJECT GUTENBERG***"
    end_marker = "*** END OF THE PROJECT GUTENBERG***"

    start = 0
    pos = text.find(start_marker)
    if pos != -1:
        start = pos + len(start_marker)
    
    end = len(text)
    pos = text.find(end_marker)
    if pos != -1:
        end = pos
    
    return text[start:end]

def basic_stats(text):

    lines = len(text.splitlines()) # Total lines
    words = len(text.split()) # Total words
    chars = len(text) # Total letters + non-letters
    letters = sum(1 for c in text if c.isalpha()) # Total letters

    return lines, words, chars, letters

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Count the relative frequency of each letter in a text file.'
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the input text file"
    )
    parser.add_argument(
        "--skip-preamble",
        action="store_true",
        help="Skip the preamble and license sections of the text"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print basic statistics of the text"
    )
    return parser.parse_args()

def count_letters():
    args = parse_arguments()
    start_time = time.time()

    with open(args.file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        if args.skip_preamble:
            text = skip_preamble(text)
        if args.stats:
            lines, words, chars, letters = basic_stats(text)
            print(f"Lines: {lines}, Words: {words}, Characters: {chars}, Letters: {letters}")
    
    uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lowercase = 'abcdefghijklmnopqrstuvwxyz'
    counts = [0] * len(uppercase)

    for line in text:
        for char in line:
            if char in uppercase:
                counts[uppercase.index(char)] += 1
            elif char in lowercase:
                counts[lowercase.index(char)] += 1
    
    letters = sum(counts)
    frequencies = [100.0 * count / letters for count in counts]

    for i in range(len(uppercase)):
            print((uppercase[i]), lowercase[i], frequencies[i])
    
    print(f"Check sum of frequencies: {sum(frequencies):.2f}%")
    
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":  # Replace with the actual file path
    count_letters()