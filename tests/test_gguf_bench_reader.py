import pytest
from tools.bench.compare_gguf_iq3s_4090 import case_metrics


@pytest.mark.parametrize('version', [16, 17, 18, 19])
def test_legacy_rates_and_observed_ttft(version):
    timings = {'prepare_seconds': 1, 'prefill_seconds': 2, 'decode_seconds': 3}
    if version == 19:
        timings['first_token_seconds'] = 3.5
    report = {'schema_version': version, 'tests': [
        {'label': 'pp512+tg128', 'prefill_tok_s_mean': 256,
         'decode_output_tok_s_mean': 128 / 3, 'total_seconds_mean': 6,
         'reps': [{'timings': timings}]}]}
    row = case_metrics(report)['pp512+tg128']
    assert row['prefill_tok_s_mean'] == 256
    assert row['decode_output_tok_s_mean'] == 128 / 3
    assert row['ttft_seconds_mean'] == (3.5 if version == 19 else None)


@pytest.mark.parametrize('version', [15, 20, None])
def test_unsupported_report_is_rejected(version):
    with pytest.raises(ValueError, match='16–19'):
        case_metrics({'schema_version': version, 'tests': []})
