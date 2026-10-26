import numpy as np

from src.common.splits import repeated_stratified
from src.eyelid.few_local import local_sample


def test_local_photos_come_from_the_training_part_and_hold_both_labels():
    y = (np.arange(122) < 23).astype(int)  # Italy: 23 of 122 anemic, the scarcest label
    for i, (_, tr, te) in enumerate(repeated_stratified(y, 5, 5, 42)):
        for k in (10, 20, 40):
            local = local_sample(y, tr, k, i)
            assert len(local) == k and set(local) <= set(tr) and set(y[local]) == {0, 1}
    assert len(local_sample(y, tr, 0, 0)) == 0 and (local_sample(y, tr, "all", 0) == tr).all()
