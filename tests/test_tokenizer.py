from phage_kmer_lm.tokenizer import KmerTokenizer


def test_roundtrip_stride_one_kmers():
    tokenizer = KmerTokenizer.build(k=3)
    sequence = "ACGTACGT"
    ids = tokenizer.encode(sequence)
    assert tokenizer.decode(ids, strict_overlap=True) == sequence


def test_compatible_transition_has_four_bases():
    tokenizer = KmerTokenizer.build(k=3)
    token_id = tokenizer.token_to_id["ACG"]
    texts = {tokenizer.id_to_token[i] for i in tokenizer.compatible_next_ids(token_id)}
    assert texts == {"CGA", "CGC", "CGG", "CGT"}
