"""Compare BGE CPU forward passes in PyTorch and ONNX Runtime."""

import argparse
import json
import math
import platform
from itertools import product
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter


def summarize_latency(latency_ms: list[float], batch_size: int) -> dict[str, float]:
    """Use the nearest-rank P95 and calculate throughput from total time."""
    ordered = sorted(latency_ms)
    return {
        "mean_ms": mean(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "texts_per_second": batch_size * len(ordered) * 1000 / sum(ordered),
    }


def measure(predict, warmups: int, repetitions: int) -> list[float]:
    for _ in range(warmups):
        predict()
    latency_ms = []
    for _ in range(repetitions):
        start = perf_counter()
        predict()
        latency_ms.append((perf_counter() - start) * 1000)
    return latency_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--token-lengths", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/linux-cpu-matrix.json"))
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or min(args.threads, args.repetitions) < 1 or args.warmups < 0:
        parser.error("Batch sizes, threads and repetitions must be positive. Warmups may be zero.")
    if any(not 2 <= length <= 512 for length in args.token_lengths):
        parser.error("Token lengths must be between 2 and 512 for this model.")
    if len(set(args.batch_sizes)) != len(args.batch_sizes) or len(set(args.token_lengths)) != len(args.token_lengths):
        parser.error("List each batch size and token length once.")
    return args


def main() -> None:
    args = parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    from lesson01_forward import DEFAULT_TEXTS, MODEL_ID, MODEL_REVISION

    class Encoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).last_hidden_state

    torch.set_num_threads(args.threads)
    model_options = {"revision": MODEL_REVISION}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **model_options)
    pretrained = AutoModel.from_pretrained(
        MODEL_ID, use_safetensors=True, attn_implementation="eager", **model_options
    )
    encoder = Encoder(pretrained).eval()
    input_names = ("input_ids", "attention_mask", "token_type_ids")
    output_name = "last_hidden_state"

    def tokenize(batch_size: int, length: int):
        texts = [DEFAULT_TEXTS[index % len(DEFAULT_TEXTS)] for index in range(batch_size)]
        return tokenizer(texts, padding="max_length", max_length=length, truncation=True, return_tensors="pt")

    example = tokenize(args.batch_sizes[0], args.token_lengths[0])
    records = []
    atol, rtol = 5e-5, 1e-4
    with TemporaryDirectory(prefix="inferbench-") as temporary_dir:
        model_path = str(Path(temporary_dir) / "bge-small-zh.onnx")
        dynamic_axes = {name: {0: "batch_size", 1: "token_length"} for name in (*input_names, output_name)}
        with torch.no_grad():
            torch.onnx.export(
                encoder, tuple(example[name] for name in input_names), model_path,
                input_names=list(input_names), output_names=[output_name],
                dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
            )
        onnx.checker.check_model(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        runtime = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])

        with torch.inference_mode():
            for batch_size, length in product(args.batch_sizes, args.token_lengths):
                inputs = tokenize(batch_size, length)
                torch_args = tuple(inputs[name] for name in input_names)
                ort_inputs = {name: inputs[name].numpy() for name in input_names}

                def predict_torch():
                    return encoder(*torch_args)

                def predict_ort():
                    return runtime.run([output_name], ort_inputs)[0]

                expected, actual = predict_torch().numpy(), predict_ort()
                shape = (batch_size, length, pretrained.config.hidden_size)
                if expected.shape != shape or actual.shape != shape:
                    raise ValueError(f"Expected output shape {shape}. Got {expected.shape} and {actual.shape}.")
                np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
                max_difference = float(np.max(np.abs(actual - expected)))

                for backend, predict in (("PyTorch", predict_torch), ("ONNX Runtime", predict_ort)):
                    samples = measure(predict, args.warmups, args.repetitions)
                    summary = summarize_latency(samples, batch_size)
                    records.append({
                        "backend": backend, "batch_size": batch_size, "token_length": length,
                        **summary, "latency_ms": samples, "max_abs_difference": max_difference,
                    })
                    print(f"{backend:12} ({batch_size}, {length})  "
                          f"P95 {summary['p95_ms']:.2f} ms  {summary['texts_per_second']:.1f} texts/s")

    result = {
        "model": MODEL_ID, "revision": MODEL_REVISION, "platform": platform.system(),
        "platform_release": platform.release(), "machine": platform.machine(), "device": "CPU",
        "pytorch_version": torch.__version__, "ort_version": ort.__version__,
        "transformers_version": transformers.__version__, "onnx_version": onnx.__version__,
        "intra_op_threads": args.threads, "warmups": args.warmups,
        "measured_batches_per_config": args.repetitions, "timer_scope": "forward_only",
        "sample_texts": list(DEFAULT_TEXTS), "atol": atol, "rtol": rtol, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} benchmark records to {args.output}")


if __name__ == "__main__":
    main()
