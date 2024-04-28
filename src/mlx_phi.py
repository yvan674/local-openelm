import json
from pathlib import Path
from typing import Generator

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.utils import generate_step
from mlx_lm.tokenizer_utils import TokenizerWrapper


class MlxPhi:
    def __init__(self,
                 model_fp: Path,
                 system_prompt: str = None):
        """Provides an object interface to talk to models.

        This class also holds message history.

        Args:
            model_fp: Path to the huggingface model repository.
            system_prompt: If desired, a system prompt for the model.
        """
        self.model, self.tokenizer = load(str(model_fp))

        if not isinstance(self.tokenizer, TokenizerWrapper):
            self.tokenizer = TokenizerWrapper(self.tokenizer)

        self.messages = []

        if system_prompt is not None:
            self.messages.append({"role": "system",
                                  "content": system_prompt})

        # One-shot to start messages template
        self.messages.append({"role": "user",
                              "content": "Are you ready to begin?"})
        self.messages.append({"role": "assistant",
                              "content": "Yes, I am ready to answer any of "
                                         "your queries."})

        # Now figure out what the added tokens were
        with open(model_fp / "added_tokens.json") as f:
            added_tokens = json.load(f)

        self.eos_tokens = ["<|end|>", "<|endoftext|>"]

        self.eos_token_ids = [added_tokens[t] for t in self.eos_tokens]

        # The default string end
        self.eos_tokens.append("</s>")
        self.eos_token_ids.append(2)


    def generate(self, prompt: str):
        """Provides a response to a prompt."""
        self.tokenizer.apply_chat_template([{"role": "user",
                                             "content": prompt}],
                                           tokenize=False)
        return generate(self.model, self.tokenizer, prompt, temp=0.4)

    def generate_stream(self,
                        prompt: str,
                        max_tokens: int = 1024) -> Generator[str, None, None]:
        """Provides a generator object for the prompt."""
        self.messages.append({"role": "user", "content": prompt})

        tokens = self.tokenizer.apply_chat_template(self.messages,
                                                    tokenize=True)
        prompt_tokens = mx.array(tokens)

        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        response = ""

        for (token, prob), n in zip(generate_step(prompt_tokens,
                                                  self.model,
                                                  temp=0.4),
                                    range(max_tokens)):
            if token in self.eos_token_ids:
                break
            detokenizer.add_token(token)
            token_word = detokenizer.last_segment
            response += token_word
            yield token_word

        self.messages.append({"role": "assistant",
                              "content": response})


if __name__ == '__main__':
    _mlx_model = MlxPhi(
        Path("/Users/Yvan/Models/Phi-3-mini-4k-instruct-4bit-no-q-embed"),
        "You are a helpful and cheerful assistant."
    )
    for _token in _mlx_model.generate_stream("Hi there, how are you?"):
        print(_token, end="", flush=True)

    # print(_mlx_model.generate("Hi there, how are you?"))
