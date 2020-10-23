import os
import re
from contextlib import contextmanager
from timeit import default_timer
from typing import Iterable, List, Tuple, Union

import numpy as np
import scipy.signal as ss
import torch.nn as nn


@contextmanager
def ElapsedTimer():
    start_time = default_timer()

    class _Timer:
        start = start_time
        end = default_timer()
        duration = end - start

    yield _Timer

    end_time = default_timer()
    _Timer.end = end_time
    _Timer.duration = end_time - start_time


class View(nn.Module):
    """Utility module for reshaping tensors within nn.Sequential block."""

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class RangeKeyDict(dict):
    """Custom 'dictionary' for reverse lookup of range"""

    def __init__(self, *args, **kwargs):
        super(RangeKeyDict, self).__init__(*args, **kwargs)

        # !any(!A or !B) is faster than all(A and B)
        assert not any(
            map(
                lambda x: not isinstance(x, tuple) or len(x) != 2 or x[0] > x[1],
                self,
            )
        )

    def __getitem__(self, number):
        try:
            result = next(
                (value for key, value in self.items() if key[0] <= number < key[1])
            )
        except StopIteration:
            raise KeyError(number)
        return result


def clean_ecg_nk2(ecg_signal, sampling_rate=500):
    """
    Parallelized version of neurokit2 ecg_clean(method="neurokit")
    ecg_signal shape should be (signal_length, number of leads)
    """
    # Remove slow drift with highpass Butterworth.
    sos = ss.butter(
        5,
        (0.5,),
        btype="highpass",
        output="sos",
        fs=sampling_rate,
    )
    clean = ss.sosfiltfilt(sos, ecg_signal, axis=0).T

    # DC offset removal with 50hz powerline filter (convolve average kernel)
    if sampling_rate >= 100:
        b = np.ones(int(sampling_rate / 50))
    else:
        b = np.ones(2)
    a = [
        len(b),
    ]
    clean = np.copy(ss.filtfilt(b, a, clean, method="pad", axis=1).T)

    return clean


def parse_comments(comments: List[str]):
    """
    Parse an PhysioNet/CinC 2020 ECG record header to get the age, sex, and SNOMEDCT codes.
    """
    age = -1
    sex = -1
    dx = []
    for comment in comments:
        dx_grp = re.search(r"Dx: (?P<dx>.*)$", comment)
        if dx_grp:
            raw_dx = dx_grp.group("dx").split(",")
            for dxi in raw_dx:
                snomed_code = int(dxi)
                dx.append(snomed_code)
            continue

        age_grp = re.search(r"Age: (?P<age>.*)$", comment)
        if age_grp:
            age = float(age_grp.group("age"))
            if not np.isfinite(age):
                age = -1
            continue

        sx_grp = re.search(r"Sex: (?P<sx>.*)$", comment)
        if sx_grp:
            if sx_grp.group("sx").upper().startswith("F"):
                sex = 1
            elif sx_grp.group("sx").upper().startswith("M"):
                sex = 0
            continue
    return age, sex, dx


def walk_files(
    root: str,
    suffix: Union[str, Tuple[str]],
    prefix: bool = False,
    remove_suffix: bool = False,
) -> Iterable[str]:
    """List recursively all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the full path to each result, otherwise
            only returns the name of the files found (Default: ``False``)
        remove_suffix (bool, optional): If true, removes the suffix to each result defined in suffix,
            otherwise will return the result as found (Default: ``False``).
    """

    root = os.path.expanduser(root)

    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        # `dirs` is the list used in os.walk function and by sorting it in-place here, we change the
        # behavior of os.walk to traverse sub directory alphabetically
        # see also
        # https://stackoverflow.com/questions/6670029/can-i-force-python3s-os-walk-to-visit-directories-in-alphabetical-order-how#comment71993866_6670926
        files.sort()
        for f in files:
            if f.endswith(suffix):

                if remove_suffix:
                    f = f[: -len(suffix)]

                if prefix:
                    f = os.path.join(dirpath, f)

                yield os.path.normpath(f)


def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


# Load a table with row and column names.
def load_weights_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, "r") as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(",")]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table) - 1
    if num_rows < 1:
        raise Exception("The table {} is empty.".format(table_file))

    num_cols = set(len(table[i]) - 1 for i in range(num_rows))
    if len(num_cols) != 1:
        raise Exception(
            "The table {} has rows with different lengths.".format(table_file)
        )
    num_cols = min(num_cols)
    if num_cols < 1:
        raise Exception("The table {} is empty.".format(table_file))

    # Find the row and column labels.
    rows = [table[0][j + 1] for j in range(num_rows)]
    cols = [table[i + 1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i + 1][j + 1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float("nan")

    return rows, cols, values
