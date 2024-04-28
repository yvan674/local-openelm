"""Hyde test.

Test to see if Hyde is at all reasonable on Phi-3
"""
from pathlib import Path
from time import perf_counter

from mlx_phi import MlxPhi


def main():

    questions = ["Please write a single sentence to answer the question\n"
                 "Question: {}\nAnswer:",
                 "Please write a sentence from a scientific paper "
                 "to support/refute the claim\n"
                 "Claim: {}\nPassage:",
                 "Please write a one-sentence counter argument\n"
                 "Argument: {}\nCounter argument:",
                 "Please write a sentence from a scientific paper to answer "
                 "the question\n"
                 "Question: {}\nPassage:",
                 "Please write a sentence from a financial article to answer "
                 "the question\n"
                 "Question: {}\nPassage",
                 "Please write a sentence from a news article about the topic\n"
                 "Question: {}\nPassage:"]

    for question in questions:
        phi = MlxPhi(Path("/Users/Yvan/Models/"
                          "Phi-3-mini-4k-instruct-4bit-no-q-embed"))
        start_time = perf_counter()
        q = question.format("Who is the CEO of AlpineAI?")
        answer = "".join([token for token in phi.generate_stream(q)])
        print(answer)


if __name__ == '__main__':
    main()
