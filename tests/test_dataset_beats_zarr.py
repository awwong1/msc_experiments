import unittest

from datasets import BeatsZarr


class BeatsZarrTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = BeatsZarr()

    def test_ds_len(self):
        self.assertEqual(len(self.ds), 801266)
