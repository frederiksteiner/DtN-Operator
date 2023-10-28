import unittest


class TestCase(unittest.TestCase):
    def test_test(self) -> None:
        a = 1
        b = 1
        self.assertEqual(a, b)
