"""Run one CPU forward pass and show how Chinese text becomes BERT tensors."""

import argparse
import platform

import torch
import transformers
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
DEFAULT_TEXTS = ("这款耳机音质很好。", "音质很好。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="*", help="Override the two example sentences.")
    parser.add_argument("--max-length", type=int, default=16, help="Padded token length (default: 16).")
    args = parser.parse_args()
    if not 2 <= args.max_length <= 512:
        parser.error("--max-length must be between 2 and 512 for this model")
    texts = args.texts or list(DEFAULT_TEXTS)

    print(f"Environment: {platform.system()} {platform.machine()}, PyTorch {torch.__version__}")
    print(f"Transformers {transformers.__version__}, device=cpu, model={MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_safetensors=True).eval()
    model_inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    print(f"input_ids shape: {tuple(model_inputs['input_ids'].shape)}")
    print(f"attention_mask shape: {tuple(model_inputs['attention_mask'].shape)}")
    for text, ids, mask in zip(texts, model_inputs["input_ids"], model_inputs["attention_mask"]):
        token_ids = ids.tolist()
        print(f"\nText: {text}")
        print(f"tokens: {tokenizer.convert_ids_to_tokens(token_ids)}")
        print(f"input_ids:      {token_ids}")
        print(f"attention_mask: {mask.tolist()}")

    # Inference needs the forward result without gradients for training.
    with torch.inference_mode():
        hidden_state = model(**model_inputs).last_hidden_state

    expected_shape = (len(texts), args.max_length, model.config.hidden_size)
    actual_shape = tuple(hidden_state.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(f"Expected hidden state shape {expected_shape}, got {actual_shape}")
    print(f"\nlast_hidden_state shape: {actual_shape}")


if __name__ == "__main__":
    main()
