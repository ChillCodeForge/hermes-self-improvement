import unittest

from durations import parse_duration


class ParseDurationTests(unittest.TestCase):
    def test_single_units(self):
        self.assertEqual(parse_duration("45s"), 45)
        self.assertEqual(parse_duration("2h"), 7200)

    def test_combined_units(self):
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("1h0m5s"), 3605)

    def test_rejects_garbage(self):
        for text in ["", "abc", "10", "5x", "1h 30m", "30m1h", "1h1h", "-5s"]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_duration(text)


if __name__ == "__main__":
    unittest.main()
