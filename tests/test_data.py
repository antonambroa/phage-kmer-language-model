from phage_kmer_lm.data import make_windows, split_examples


def test_fixed_context_windows():
    windows = make_windows(list(range(20)), context_length=5, stride=3)
    assert all(len(w) == 6 for w in windows)
    assert windows[-1][-1] == 19


def test_split_is_deterministic():
    examples = [[i, i + 1] for i in range(20)]
    assert split_examples(examples, 0.2, 42) == split_examples(examples, 0.2, 42)
