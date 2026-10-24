import numpy as np
from PIL import Image

from src.nails import load_nails, nail_pixels, white_balance


def test_white_balance_makes_the_board_white():
    rgb = np.full((10, 10, 3), [0.8, 0.9, 0.9], dtype=np.float32)  # a bluish white board
    rgb[:2, :2] = [0.4, 0.3, 0.3]  # a nail: too dark and too colored to count as board
    out = white_balance(rgb)
    assert np.allclose(out[5, 5], 1.0)
    assert np.allclose(out[0, 0], [0.5, 0.3 / 0.9, 0.3 / 0.9])


def test_nail_boxes_are_row_col(tmp_path):
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    img[5:10, 25:35] = [255, 0, 0]  # rows 5-9, columns 25-34
    path = tmp_path / "x.png"
    Image.fromarray(img).save(path)
    px = nail_pixels(path, [[5, 25, 10, 35]])
    assert px.shape == (50, 3)
    assert np.allclose(px, [1, 0, 0])


def test_cutoff_is_strictly_below_and_in_g_per_l(tmp_path):
    (tmp_path / "metadata.csv").write_text(
        "PATIENT_ID,MEASUREMENT_DATE,HB_LEVEL_GperL,NAIL_BOUNDING_BOXES,SKIN_BOUNDING_BOXES\n"
        '1,s1,119.0,"[[0, 0, 1, 1]]","[]"\n'
        '2,s1,120.0,"[[0, 0, 1, 1]]","[]"\n'
    )
    df = load_nails({"data_dir": str(tmp_path), "anemia_cutoff_g_per_l": 120})
    assert list(df.anemic) == [1, 0]
