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
from collections import Counter
import string
import time

# def skip_preamble(file_path):

def basic_stats(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    num_lines = len(text.splitlines()) # Total lines
    num_words = len(text.split()) # Total words
    num_chars = len(text) # Total letters + non-letters
    letters = sum(1 for c in text if c.isalpha()) # Total letters

    return num_lines, num_words, num_chars, letters

# def parse_arguments():

def count_letters(file_path):

    uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lowercase = 'abcdefghijklmnopqrstuvwxyz'

    counts = [0] * len(uppercase)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            for char in line:
                if char in uppercase:
                    counts[uppercase.index(char)] += 1
                elif char in lowercase:
                    counts[lowercase.index(char)] += 1
    
    num_lines, num_words, num_chars, letters = basic_stats(file_path)
    frequencies = counts / letters

    for i in range(len(uppercase)):
            print((uppercase[i]), lowercase[i], frequencies[i])

if __name__ == "__main__":
    count_letters()