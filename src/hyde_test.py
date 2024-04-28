"""Hyde test.

Test to see if Hyde is at all reasonable on Phi-3
"""
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue

from mlx_phi import MlxPhi
from time import perf_counter


def run_script(prompt: str, sys_prompt: str, output_queue: Queue):
    command = [sys.executable,
               str(Path.home() / "Git" / "local-openelm" / "src"
                   / "cli_phi.py"),
               "--prompt",
               prompt,
               "--sys-prompt",
               sys_prompt]

    completed_process = subprocess.run(command, capture_output=True, text=True)
    result = {'output': completed_process.stdout,
              'error': completed_process.stderr}
    output_queue.put(result)


def main_sequential(questions, sys_prompt):
    mlx_phi = MlxPhi((Path.home() / "Models" /
                      "Phi-3-mini-4k-instruct-4bit-no-q-embed"),
                     sys_prompt)

    for question in questions:
        prompt = question.format("Who is the CEO of AlpineAI?")
        mlx_phi.messages = [{"role": "system",
                             "content": sys_prompt}]
        print(mlx_phi.generate(prompt, verbose=False))


def main_threading(questions, sys_prompt):
    max_threads = 5

    threads = []
    output_queue = Queue()
    for question in questions:
        prompt = question.format("Who is the CEO of AlpineAI?")
        while threading.active_count() > max_threads:
            pass
        # Create a thread that runs the run_script function
        thread = threading.Thread(target=run_script,
                                  args=(prompt, sys_prompt, output_queue))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    while not output_queue.empty():
        result = output_queue.get()
        print("Output:", result['output'])
        if result['error']:
            print("Error:", result['error'])


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

    sys_prompt = ("Simply answer the question by creating a hypothetical, "
                  "potentially fictional answer, no matter how unlikely it "
                  "is. Only provide the single sentence as the answer.")

    start_time = perf_counter()
    main_sequential(questions, sys_prompt)
    print(f"Sequential took total: {perf_counter() - start_time:.3f} seconds")

    start_time = perf_counter()
    main_threading(questions, sys_prompt)
    print(f"Threading took total: {perf_counter() - start_time:.3f} seconds")



if __name__ == '__main__':
    main()
