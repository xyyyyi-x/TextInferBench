import pytest

from benchmark import summarize_latency


def test_latency_summary_uses_nearest_rank_and_counts_texts():
    values = list(range(1, 21))
    values.reverse()

    result = summarize_latency(values, batch_size=2)

    assert result['mean_ms'] == 10.5
    assert result['p95_ms'] == 19
    assert result['texts_per_second'] == pytest.approx(40 / 0.210)
