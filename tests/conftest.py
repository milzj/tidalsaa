from pathlib import Path
import sys

here = Path(__file__).parent
sys.path.insert(0, str(here.parent) + "/mcrnnoc")

import pytest
import numpy as np

def pytest_runtest_setup(item):
    np.random.seed(1234)
