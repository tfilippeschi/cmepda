# Same thing as count_letters.py, but made during lecture with prof

import argparse
import time
import string

def process_file(file_path):
    start_time = time.time()
    # count_dict = {}
    # for character in string.ascii_lowercase:
    #     count_dict[character] = 0
    count_dict = {character: 0 for character in string.ascii_lowercase}
    with open(file_path, encoding="utf-8") as input_file:
        for line in input_file:
            for character in line.lower():
                try:
                    if character in count_dict:
                        count_dict[character] += 1
                except KeyError:
                    pass
    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time:.3f} s")
    print(count_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    args = parser.parse_args()
    process_file(args.file_path)

