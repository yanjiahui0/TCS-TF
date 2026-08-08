import numpy as np

from tcstf.statistics import clopper_pearson, paired_block_bootstrap_interval


def test_clopper_pearson_contains_empirical_rate():
    lo, hi = clopper_pearson(9060, 10000)
    assert lo < 0.906 < hi


def test_paired_block_bootstrap_returns_ordered_interval():
    rng = np.random.default_rng(0)
    d = rng.normal(-0.1, 0.2, 200)
    point, lo, hi = paired_block_bootstrap_interval(d, block_length=10, reps=200, seed=1)
    assert lo <= point <= hi
