import unittest

import numpy as np

import classification


class TestROIAlgorithms(unittest.TestCase):
    def setUp(self):
        self.ref_roi = np.full((32, 32), 120, dtype=np.uint8)
        self.empty_roi = self.ref_roi.copy()
        self.occupied_roi = self.ref_roi.copy()
        self.occupied_roi[8:24, 8:24] = 220

    def test_has_multiple_classifier_options(self):
        self.assertGreaterEqual(len(classification.CLASSIFIER_OPTIONS), 8)

    def test_all_methods_mark_identical_roi_as_free(self):
        for key, _label in classification.CLASSIFIER_OPTIONS:
            result = classification.classify_roi(self.ref_roi, self.empty_roi, key)
            self.assertIn("occupied", result)
            self.assertFalse(result["occupied"], msg=f"Method should mark free: {key}")

    def test_ensemble_detects_strong_center_change(self):
        result = classification.classify_roi(self.ref_roi, self.occupied_roi, "roi_ensemble")
        self.assertTrue(result["occupied"])


if __name__ == "__main__":
    unittest.main()
