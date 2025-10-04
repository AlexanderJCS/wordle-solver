"""
Calculates the guess distribution of the wordle bot
"""

import numpy as np


from main import Wordle
from tqdm import tqdm
from datetime import datetime


def main():
    with open("solution_words.txt", "r") as f:
        solution_words = f.read().splitlines()
    
    with open("dictionary.txt") as f:
        dictionary = f.read().splitlines()
    
    distribution = np.zeros((7,), dtype=np.uint64)
    
    for solution_word in tqdm(solution_words):
        w = Wordle(solution_word, dictionary)
        w.make_guess("tares")  # first guess is always tares
        
        while len(w.guesses) < w.max_attempts and not w.is_solved():
            word, entropy = w.best_word(progress=False)
            if word is None:
                break  # no possible words left
            
            w.make_guess(word)
        
        if len(w.guesses) == w.max_attempts and not w.is_solved():  # failed
            distribution[6] += 1
        else:
            distribution[len(w.guesses) - 1] += 1
    
    print("Distribution", distribution)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"distribution_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(str(distribution))
    
    indices = np.arange(len(distribution))
    weighted_avg = np.average(indices, weights=distribution) + 1
    print("Average", weighted_avg)


if __name__ == "__main__":
    main()
