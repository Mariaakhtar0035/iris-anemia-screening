import numpy as np
from PIL import Image

from src.eyelid.quality import badness, measure

CFG = {"glare": {"min_value": 0.90, "max_saturation": 0.20}, "dark_max_value": 0.15, "blown_min": 0.99,
       "blur_erode_px": 5, "min_pixels": 50}


def test_glare_pixels_are_counted_and_left_out_of_the_color_numbers(tmp_path):
    img = np.zeros((40, 40, 4), np.uint8)
    img[:, :, 3] = 255
    img[:, :, :3] = (200, 80, 90)          # pink tissue
    img[:4, :10, :3] = 255                 # 40 of 1600 pixels: a white highlight
    Image.fromarray(img).save(tmp_path / "cut.png")
    q, plain, no_glare = measure(tmp_path / "cut.png", CFG, 255)
    assert np.isclose(q["glare"], 40 / 1600) and q["size"] == 1600
    assert no_glare["a_mean"] > plain["a_mean"]          # without the highlight the region reads redder
    assert np.isclose(no_glare["R_mean"], 200 / 255, atol=1e-3)


def test_badness_ranks_the_worst_photo_on_any_measure_last():
    q = {"glare": np.array([0.0, 0.0, 0.5]), "dark": np.zeros(3), "blown_out": np.zeros(3),
         "blur": np.array([10.0, 1.0, 10.0])}
    assert badness(q).argmax() in (1, 2) and badness(q)[0] < badness(q)[1]
