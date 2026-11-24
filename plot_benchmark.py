"""Turn InferBench results into a P95 latency chart."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

# Render to a file on headless machines as well as on a local desktop.
matplotlib.use("Agg")
from matplotlib import pyplot as plt


BACKENDS = ("PyTorch", "ONNX Runtime")


def paired_latencies(records: list[dict]) -> dict[tuple[int, int], dict[str, float]]:
    """Group P95 values by input shape and keep shapes with both backends."""
    groups: dict[tuple[int, int], dict[str, float]] = {}
    for record in records:
        backend = record["backend"]
        if backend not in BACKENDS:
            continue

        shape = (record["batch_size"], record["token_length"])
        latencies = groups.setdefault(shape, {})
        if backend in latencies:
            raise ValueError(f"Duplicate {backend} result for shape {shape}.")

        p95 = record["p95_ms"]
        if not isinstance(p95, (int, float)) or not math.isfinite(p95) or p95 < 0:
            raise ValueError(f"Invalid P95 latency for {backend} at shape {shape}.")
        latencies[backend] = p95

    return {shape: values for shape, values in groups.items() if len(values) == len(BACKENDS)}


def save_latency_plot(result: dict, output: Path) -> None:
    groups = paired_latencies(result["records"])
    if not groups:
        raise ValueError("No paired shapes with both PyTorch and ONNX Runtime results.")

    shapes = sorted(groups)
    positions = range(len(shapes))
    labels = [f"Batch {batch_size}\n{token_length} tokens" for batch_size, token_length in shapes]
    colors = ("#215587", "#C16B29")
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(max(7.0, 1.65 * len(shapes)), 4.8), layout="constrained")
    try:
        for index, (backend, color) in enumerate(zip(BACKENDS, colors)):
            offset = (index - 0.5) * bar_width
            bars = axis.bar(
                [position + offset for position in positions],
                [groups[shape][backend] for shape in shapes],
                width=bar_width,
                color=color,
                label=backend,
            )
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

        platform = result.get("platform", "Unknown platform")
        device = result.get("device", "Unknown device")
        figure.suptitle(f"P95 inference latency\n{platform} / {device}", fontsize=15, fontweight="bold")
        axis.set_xticks(positions, labels)
        axis.set_ylabel("P95 latency (ms)")
        axis.margins(y=0.16)
        axis.set_ylim(bottom=0)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#DCE3EA", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)

        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="png", dpi=160)
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Benchmark result JSON file")
    parser.add_argument("--output", type=Path, required=True, help="PNG file to write")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")

    try:
        result = json.loads(args.input.read_text(encoding="utf-8-sig"))
        save_latency_plot(result, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    print(f"Chart saved to {args.output}")


if __name__ == "__main__":
    main()
