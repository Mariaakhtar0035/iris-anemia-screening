import numpy as np
import pandas as pd
import pytest

from src.eyelid.metadata import age_sex


def test_age_and_sex_read_both_datasets_and_reject_unknown_sex():
    ghana = pd.DataFrame({"age_months": [6, 24], "sex": ["Male", "Female"]})
    adults = pd.DataFrame({"age": [30, 70], "sex": ["F", "M"]})
    assert (age_sex(ghana) == np.array([[6, 1], [24, 0]])).all()
    assert (age_sex(adults) == np.array([[30, 0], [70, 1]])).all()
    with pytest.raises(ValueError):
        age_sex(pd.DataFrame({"age": [30], "sex": ["X"]}))
