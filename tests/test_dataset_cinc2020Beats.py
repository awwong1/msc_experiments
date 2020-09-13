import unittest
# from torch.utils.data import DataLoader
from datasets import CinC2020Beats


class CinC2020BeatsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds_beat = CinC2020Beats(pqrst_window_size=400)
        ds_beat_300 = CinC2020Beats(pqrst_window_size=300)
        ds_beat_500 = CinC2020Beats(pqrst_window_size=500)

    def test_len(self):
        self.assertEqual(len(self.ds_beat), 767097)

    def test_getitem(self):
        window, age, sex, dx = self.ds_beat[0]
        self.assertEqual(window.shape, (400, 12))
        self.assertEqual(age, 53.0)
        self.assertEqual(sex, 0.0)
        # self.assertEqual(dx, [164867002, 427084000])
        self.assertEqual(dx, 164867002)

        window, age, sex, dx = self.ds_beat[len(self.ds_beat) - 1]
        self.assertEqual(window.shape, (400, 12))
        self.assertEqual(age, 61.0)
        self.assertEqual(sex, 1.0)
        self.assertEqual(dx, 164865005)
