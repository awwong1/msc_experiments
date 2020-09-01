import unittest
from datasets import CinC2020


class CinC2020TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds_varlen = CinC2020()
        cls.ds_setlen = CinC2020(set_seq_len=5000)
        cls.ds_setbiglen = CinC2020(set_seq_len=20000)

    def test_len(self):
        self.assertEqual(len(self.ds_varlen), 43101)
        self.assertEqual(len(self.ds_setlen), 71711)
        self.assertEqual(len(self.ds_setbiglen), 47828)

    def test_getitem(self):
        # first
        p_signal, sampling_rate, age, sex, dx = self.ds_varlen[0]
        self.assertEqual(p_signal.shape, (5000, 12))
        self.assertEqual(sampling_rate, 500)
        self.assertEqual(age, 53.0)
        self.assertEqual(sex, 0.0)
        # self.assertEqual(dx, [164867002, 427084000])
        self.assertEqual(dx, 164867002)

        # last
        p_signal, sampling_rate, age, sex, dx = self.ds_varlen[43100]
        self.assertEqual(p_signal.shape, (60006, 12))
        self.assertEqual(sampling_rate, 500)
        self.assertEqual(age, 61.0)
        self.assertEqual(sex, 1.0)
        # self.assertEqual(dx, [164865005, ])
        self.assertEqual(dx, 164865005)

        # first
        p_signal, sampling_rate, age, sex, dx = self.ds_setlen[0]
        self.assertEqual(p_signal.shape, (5000, 12))
        self.assertEqual(sampling_rate, 500)
        self.assertEqual(age, 53.0)
        self.assertEqual(sex, 0.0)
        # self.assertEqual(dx, [164867002, 427084000])
        self.assertEqual(dx, 164867002)

        # last
        p_signal, sampling_rate, age, sex, dx = self.ds_setlen[71710]
        self.assertEqual(p_signal.shape, (5000, 12))
        self.assertEqual(sampling_rate, 500)
        self.assertEqual(age, 61.0)
        self.assertEqual(sex, 1.0)
        # self.assertEqual(dx, [164865005, ])
        self.assertEqual(dx, 164865005)

        # first
        p_signal, sampling_rate, age, sex, dx = self.ds_setbiglen[0]
        self.assertEqual(p_signal.shape, (20000, 12))
        self.assertEqual(sampling_rate, 500)
        self.assertEqual(age, 53.0)
        self.assertEqual(sex, 0.0)
        # self.assertEqual(dx, [164867002, 427084000])
        self.assertEqual(dx, 164867002)

        # last
        p_signal, sampling_rate, age, sex, dx = self.ds_setbiglen[47827]
        self.assertEqual(p_signal.shape, (20000, 12))
        self.assertEqual(sampling_rate, 500)
        self.assertEqual(age, 61.0)
        self.assertEqual(sex, 1.0)
        # self.assertEqual(dx, [164865005, ])
        self.assertEqual(dx, 164865005)

    def test_len_data_cache(self):
        self.assertCountEqual(self.ds_varlen.len_data.keys(), self.ds_varlen.ecg_records)
        self.assertCountEqual(self.ds_setlen.len_data.keys(), self.ds_setlen.ecg_records)
        self.assertCountEqual(self.ds_setbiglen.len_data.keys(), self.ds_setbiglen.ecg_records)
