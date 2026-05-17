#!/usr/bin/env python
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_stman import split_raw_data


class TrainDataSplitTests(unittest.TestCase):
    def test_split_raw_data_uses_explicit_train_and_validation_ratios(self):
        raw = np.zeros((10, 2, 1), dtype=np.float32)

        train, val, test = split_raw_data(raw, split_rate=0.7, val_ratio=0.1)

        self.assertEqual([part.shape[0] for part in (train, val, test)], [7, 1, 2])


if __name__ == "__main__":
    unittest.main()
