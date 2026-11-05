import numpy as np

from src.eyelid.auto_crop import dice, grid, patch_share, to_mask


def test_to_mask_keeps_the_largest_piece_and_flags_not_found():
    prob = np.zeros((4, 4))
    prob[0, 0] = 0.9                    # a one-patch speck
    prob[2:4, 1:4] = 0.8                # the eyelid: six patches
    mask, found, conf = to_mask(prob, (40, 40), 0.5)
    assert found and not mask[0:5, 0:5].any() and mask[30, 25] and 0.5 < conf <= 0.8
    _, found, _ = to_mask(np.full((4, 4), 0.2), (40, 40), 0.5)
    assert not found


def test_dice_patch_share_and_grid():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    assert dice(a, a) == 1.0 and dice(a, ~a) == 0.0
    assert np.allclose(patch_share(a, 2, 1), [1.0, 0.0])
    assert grid((600, 800), 448, 14) == (24, 32) and grid((800, 600), 448, 14) == (32, 24)
