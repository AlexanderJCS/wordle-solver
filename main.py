from dataclasses import dataclass
from enum import Enum
from math import log2
from typing import Optional

from tqdm import tqdm


class GuessResult(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    GRAY = "gray"


@dataclass(frozen=True)
class GuessedLetter:
    letter: str
    result: GuessResult


@dataclass(frozen=True)
class Guess:
    letters: list[GuessedLetter]


class Wordle:
    def __init__(self, solution: str, dictionary: list[str]):
        self.solution = solution
        self.dictionary = dictionary
        self.guesses: list[Guess] = []
        self.max_attempts = 6

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

    def possible_words(self) -> list[str]:
        possible = []
        for word in self.dictionary:
            if len(word) != len(self.solution):
                continue
            if all(self._matches_guess(word, guess) for guess in self.guesses):
                possible.append(word)
        
        return possible

    def entropy_of_guess(self, guess_word: str) -> float:
        pattern_counts = {}

        for possible_word in self.possible_words():
            simulated_guess = self.suppose_guess(possible_word, guess_word)
            pattern = tuple(letter.result for letter in simulated_guess.letters)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        total_possible = len(self.possible_words())
        entropy = 0.0
        for count in pattern_counts.values():
            probability = count / total_possible
            entropy -= probability * log2(probability)

        return entropy

    def best_word(self) -> Optional[str]:
        best_entropy = -1.0
        best_word = None

        for word in tqdm(self.dictionary):
            if len(word) != len(self.solution):
                continue
            entropy = self.entropy_of_guess(word)
            if entropy > best_entropy:
                best_entropy = entropy
                best_word = word

        return best_word
    

def main():
    with open("dictionary.txt") as f:
        dictionary = f.read().splitlines()
    
    w = Wordle("beats", dictionary)
    w.make_guess("beaut")

    print(w.best_word())


if __name__ == "__main__":
    main()
