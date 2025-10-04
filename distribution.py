"""
Calculates the guess distribution of the wordle bot
"""

import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

from main import Wordle
from tqdm import tqdm
from datetime import datetime


def solve_word(solution_word, dictionary):
    """Worker function to solve a single word and return the number of guesses used"""
    w = Wordle(solution_word, dictionary)
    w.make_guess("tares")  # first guess is always tares
    
    while len(w.guesses) < w.max_attempts and not w.is_solved():
        word, entropy = w.best_word(progress=False)
        if word is None:
            break  # no possible words left
        
        w.make_guess(word)
    
    if len(w.guesses) == w.max_attempts and not w.is_solved():  # failed
        return 6  # failed case
    else:
        return len(w.guesses) - 1  # successful case (0-indexed)


def main():
    with open("solution_words.txt", "r") as f:
        solution_words = f.read().splitlines()
    
    with open("dictionary.txt") as f:
        dictionary = f.read().splitlines()
    
    distribution = np.zeros((7,), dtype=np.uint64)
    
    # Use multiprocessing with progress bar
    num_processes = cpu_count()
    print(f"Using {num_processes} processes")
    
    # Create partial function with dictionary pre-loaded
    worker_func = partial(solve_word, dictionary=dictionary)
    
    # Use Pool with tqdm for progress tracking
    with Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(pool.imap(worker_func, solution_words),
                          total=len(solution_words),
                          desc="Solving words"):
            results.append(result)
    
    # Count results into distribution
    for result in results:
        distribution[result] += 1
    
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
