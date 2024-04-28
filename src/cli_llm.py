from argparse import ArgumentParser
from pathlib import Path

from mlx_phi import MlxPhi


model_mapping = {
    "Phi-3-mini-128k": "Phi-3-mini-128k-instruct-4bit",
    "Phi-3-mini-4k": "Phi-3-mini-4k-instruct-4bit-no-q-embed"
}


def parse_args():
    p = ArgumentParser(description="A CLI to a local LLM.")

    p.add_argument("--model-name", type=str,
                   help="The model to use.", default=None,
                   choices=["Phi-3-mini-128k", "Phi-3-mini-4k"])

    return p.parse_args()



def model_to_fp(model_name: str) -> Path:
    """Turns a model identifier into a Path where the model is stored.

    Currently supports:
        - Phi-3-mini-128k
        - Phi-3-mini-4k

    Args:
        model_name: One of the supported model names listed above.

    Returns:
        Path to the model dir in the ~/Models/ dir
    """
    out_path = Path.home() / "Models"
    if model_name in model_mapping:
        return out_path / model_mapping[model_name]
    else:
        raise ValueError(f"{model_name=} is not one of the supported models.")


def choose_model():
    model_int_map = {i: model_name
                     for i, model_name in enumerate(model_mapping.keys())}
    for i, model_name in model_int_map.items():
        print(f"{i}: {model_name}")

    model_idx = int(input("Choose a model: "))
    return model_int_map[model_idx]


def main(model_name: str = None):
    if model_name is None:
        model_name = choose_model()
    model_fp = model_to_fp(model_name)
    mlx_model = MlxPhi(model_fp, "You are a helpful, cheerful "
                                 "assistant.")
    print("Model loaded. Type your prompt.")
    while True:
        prompt = input(">>> ")
        if prompt == "exit" or prompt == "q" or prompt == "quit":
            break
        for token in mlx_model.generate_stream(prompt):
            print(token, end="", flush=True)
        print()


if __name__ == '__main__':
    args = parse_args()
    main(args.model_name)
