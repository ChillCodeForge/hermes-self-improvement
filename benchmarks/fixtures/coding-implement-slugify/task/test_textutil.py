import unittest

from textutil import slugify


class SlugifyTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_collapses_separators(self):
        self.assertEqual(slugify("  Release   notes -- v2.0  "), "release-notes-v2-0")

    def test_accents(self):
        self.assertEqual(slugify("Crème brûlée à la carte"), "creme-brulee-a-la-carte")

    def test_max_length_has_no_trailing_hyphen(self):
        self.assertEqual(slugify("alpha beta gamma", max_length=11), "alpha-beta")

    def test_only_symbols(self):
        self.assertEqual(slugify("!!!"), "")


if __name__ == "__main__":
    unittest.main()
