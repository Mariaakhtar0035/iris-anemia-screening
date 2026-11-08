import numpy as np
from sklearn.linear_model import LogisticRegression

from src.eyelid.tabpfn_image import with_image


def test_with_image_keeps_color_columns_and_squeezes_the_image_ones():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2 + 20))
    y = (X[:, 0] > 0).astype(int)
    model = with_image(LogisticRegression, 2, 3, 0)().fit(X, y)
    out = model[:-1].transform(X)
    assert out.shape == (60, 5)
    assert np.allclose(out[:, :2], X[:, :2])          # color numbers pass through unchanged
    assert model.predict_proba(X).shape == (60, 2)
    alone = with_image(LogisticRegression, 0, 3, 0)().fit(X, y)
    assert alone[:-1].transform(X).shape == (60, 3)   # image features alone
