"""Unit checks for weak keyword metrics; these do not validate clinical accuracy."""

import unittest

from src.evaluate import extract_labels


class WeakKeywordExtractionTests(unittest.TestCase):
    def test_negated_finding_is_not_marked_present(self):
        labels = extract_labels("No evidence of pleural effusion.")
        self.assertEqual(labels[2], 0)

    def test_contrast_resets_negation_scope(self):
        labels = extract_labels("No pleural effusion, but a focal opacity is present.")
        self.assertEqual(labels[2], 0)
        self.assertEqual(labels[3], 1)

    def test_unmentioned_categories_do_not_become_normal(self):
        labels = extract_labels("The mediastinal contour is within normal limits.")
        self.assertEqual(labels[0], 1)
        blank = extract_labels("Findings were not supplied.")
        self.assertEqual(blank.sum(), 0)


if __name__ == "__main__":
    unittest.main()
