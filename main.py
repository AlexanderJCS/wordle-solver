from dataclasses import dataclass
from enum import Enum
from math import log2
from typing import Optional

from tqdm import tqdm

import colorama
import numpy as np

import pickle
import os

colorama.init(autoreset=True)


class GuessResult(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    GRAY = "gray"


@dataclass(frozen=True)
class GuessedLetter:
    letter: str
    result: GuessResult
    
    def __str__(self):
        cmap = {
            GuessResult.GREEN: colorama.Back.GREEN + colorama.Fore.BLACK,
            GuessResult.YELLOW: colorama.Back.YELLOW + colorama.Fore.BLACK,
            GuessResult.GRAY: colorama.Back.WHITE + colorama.Fore.BLACK,
        }
        
        return f"{cmap[self.result]} {self.letter.upper()} {colorama.Style.RESET_ALL}"


@dataclass(frozen=True)
class Guess:
    letters: list[GuessedLetter]
    
    def __str__(self):
        display = ""
        
        for letter in self.letters:
            display += str(letter)
            
        return display
    
    def pattern(self):
        return tuple(letter.result for letter in self.letters)


class Wordle:
    def __init__(self, solution: str, dictionary: list[str], cache_loc="pattern_lut_cache.pickle"):
        self.solution = solution
        self.dictionary = dictionary
        self.guesses: list[Guess] = []
        self.max_attempts = 6
        self._possible_cache: Optional[list[tuple[int, str]]] = None
        
        if os.path.exists(cache_loc):
            with open(cache_loc, "rb") as f:
                self.pattern_lut = pickle.load(f)
        else:
            self.pattern_lut = self.gen_pattern_lut()
            with open(cache_loc, "wb") as f:
                pickle.dump(self.pattern_lut, f)

    def suppose_guess(self, solution: str, guess_word: str):
        if len(self.guesses) >= self.max_attempts:
            raise ValueError("Maximum number of attempts reached.")
        if len(guess_word) != len(solution):
            raise ValueError("Guess word length does not match solution length.")
        
        result: list[Optional[GuessedLetter]] = [None for _ in range(len(guess_word))]
        solution_chars = list(solution)
        guess_chars = list(guess_word)

        # First pass for GREEN
        for i in range(len(guess_chars)):
            if guess_chars[i] == solution_chars[i]:
                result[i] = GuessedLetter(guess_chars[i], GuessResult.GREEN)
                solution_chars[i] = None  # Mark as used
                
        # Second pass for YELLOW and GRAY
        for i in range(len(guess_chars)):
            if result[i] is None:
                if guess_chars[i] in solution_chars:
                    result[i] = GuessedLetter(guess_chars[i], GuessResult.YELLOW)
                    solution_chars[solution_chars.index(guess_chars[i])] = None  # Mark as used
                else:
                    result[i] = GuessedLetter(guess_chars[i], GuessResult.GRAY)
        
        return Guess(result)

    def make_guess(self, guess_word: str):
        self.guesses.append(self.suppose_guess(self.solution, guess_word))
        self._possible_cache = None

    def is_solved(self) -> bool:
        return any(all(letter.result == GuessResult.GREEN for letter in guess.letters) for guess in self.guesses)

    @staticmethod
    def _matches_guess(word: str, guess: Guess) -> bool:
        # Simulate feedback using the same two-pass algorithm as suppose_guess,
        # so repeated-letter cases are handled identically.
        word_chars = list(word)
        # first pass for GREEN
        result = [None] * len(guess.letters)
        for i, guessed_letter in enumerate(guess.letters):
            if word_chars[i] == guessed_letter.letter:
                result[i] = GuessResult.GREEN
                word_chars[i] = None

        # second pass for YELLOW / GRAY
        for i, guessed_letter in enumerate(guess.letters):
            if result[i] is None:
                if guessed_letter.letter in word_chars:
                    result[i] = GuessResult.YELLOW
                    word_chars[word_chars.index(guessed_letter.letter)] = None
                else:
                    result[i] = GuessResult.GRAY

        # compare simulated result to stored guess result
        return all(result[i] == guess.letters[i].result for i in range(len(result)))

    def possible_words(self) -> list[tuple[int, str]]:
        if self._possible_cache is not None:
            return self._possible_cache
        
        possible = []
        for idx, word in enumerate(self.dictionary):
            if len(word) != len(self.solution):
                continue
            if all(self._matches_guess(word, guess) for guess in self.guesses):
                possible.append((idx, word))
        
        self._possible_cache = possible
        
        return possible

    @staticmethod
    def pattern_index(pattern: tuple[GuessResult, ...]) -> int:
        index = 0
        for i, result in enumerate(pattern):
            multiplier = 0
            if result == GuessResult.YELLOW:
                multiplier = 1
            elif result == GuessResult.GREEN:
                multiplier = 2
            else:
                continue

            index += (3 ** i) * multiplier
        return index

    def gen_pattern_lut(self):
        lut = np.zeros((len(self.dictionary), len(self.dictionary)), dtype=np.uint8)
        
        for i, solution_word in enumerate(tqdm(self.dictionary)):
            for j, guess_word in enumerate(self.dictionary):
                guess_result = self.suppose_guess(solution_word, guess_word)
                lut[i][j] = self.pattern_index(guess_result.pattern())
        
        return lut

    def entropy_of_guess(self, guess_idx: int) -> float:
        patterns = [0 for _ in range(3 ** len(self.solution))]

        for word_idx, _ in self.possible_words():
            patterns[self.pattern_lut[word_idx][guess_idx]] += 1

        total_possible = len(self.possible_words())
        entropy = 0.0
        for count in patterns:
            if count == 0:
                continue

            probability = count / total_possible
            entropy -= probability * log2(probability)

        return entropy

    def best_word(self) -> tuple[Optional[str], float]:
        best_entropy = -1.0
        best_word = None

        for word_idx, word in enumerate(tqdm(self.dictionary)):
            if len(word) != len(self.solution):
                continue
            
            entropy = self.entropy_of_guess(word_idx)
            if entropy > -1:
                print(entropy, word)
            if entropy > best_entropy:
                best_entropy = entropy
                best_word = word

        return best_word, best_entropy
    
    def __str__(self):
        display = ""
        
        for i, guess in enumerate(self.guesses):
            display += str(guess)
            if i < len(self.guesses) - 1:
                display += "\n"

        return display


def main():
    with open("dictionary.txt") as f:
        dictionary = f.read().splitlines()
    
    w = Wordle("civil", dictionary)

    # print("Best starting word:", w.best_word())

    # w.make_guess("tares")
    w.make_guess("tares")
    w.make_guess("colin")
    w.make_guess("aahed")
    while len(w.guesses) < w.max_attempts and not w.is_solved():
        word, entropy = w.best_word()
        if word is None:
            print("No possible words left!")
            break

        print(f"Guess {len(w.guesses) + 1}/{w.max_attempts}: {word} (entropy: {entropy:.3f} bits)")
        w.make_guess(word)

    print(str(w))


if __name__ == "__main__":
    main()
