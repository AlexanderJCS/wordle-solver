from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
    
    def make_guess(self, guess_word: str):
        if len(self.guesses) >= self.max_attempts:
            raise ValueError("Maximum number of attempts reached.")
        if len(guess_word) != len(self.solution):
            raise ValueError("Guess word length does not match solution length.")
        
        result: list[Optional[GuessedLetter]] = [None for _ in range(len(guess_word))]
        solution_chars = list(self.solution)
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
        
        self.guesses.append(Guess(result))
    
    def is_solved(self) -> bool:
        return any(all(letter.result == GuessResult.GREEN for letter in guess.letters) for guess in self.guesses)
    
    def _matches_guess(self, word: str, guess: Guess) -> bool:
        word_chars = list(word)
        
        for i, guessed_letter in enumerate(guess.letters):
            if guessed_letter.result == GuessResult.GREEN:
                if word_chars[i] != guessed_letter.letter:
                    return False
                word_chars[i] = None
            elif guessed_letter.result == GuessResult.YELLOW:
                if guessed_letter.letter not in word_chars or word_chars[i] == guessed_letter.letter:
                    return False
                word_chars[word_chars.index(guessed_letter.letter)] = None
            elif guessed_letter.result == GuessResult.GRAY:
                if guessed_letter.letter in word_chars:
                    return False
    
    def possible_words(self) -> list[str]:
        possible = []
        for word in self.dictionary:
            if len(word) != len(self.solution):
                continue
            if all(self._matches_guess(word, guess) for guess in self.guesses):
                possible.append(word)
    

def main():
    with open("dictionary.txt") as f:
        dictionary = f.read().splitlines()
    
    print(Wordle("beats", dictionary).make_guess("baees"))


if __name__ == "__main__":
    main()
