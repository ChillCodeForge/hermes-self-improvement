import unittest

from paginate import page_count, paginate


class PaginateTests(unittest.TestCase):
    def test_first_page_starts_at_first_item(self):
        self.assertEqual(paginate(list(range(10)), 1, 3), [0, 1, 2])

    def test_last_partial_page(self):
        self.assertEqual(paginate(list(range(10)), 4, 3), [9])

    def test_page_beyond_end_is_empty(self):
        self.assertEqual(paginate(list(range(10)), 5, 3), [])

    def test_page_count_rounds_up(self):
        self.assertEqual(page_count(10, 3), 4)
        self.assertEqual(page_count(9, 3), 3)
        self.assertEqual(page_count(0, 3), 0)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            paginate([1], 0, 3)
        with self.assertRaises(ValueError):
            page_count(3, 0)


if __name__ == "__main__":
    unittest.main()
