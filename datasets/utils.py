import os
import re
from typing import Iterable, List, Tuple, Union

import scipy.signal as ss
import numpy as np


def parse_comments(comments: List[str]):
    """
    Parse an PhysioNet/CinC 2020 ECG record header to get the age, sex, and SNOMEDCT codes.
    """
    age = float("nan")
    sex = float("nan")
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
                age = float("nan")
            continue

        sx_grp = re.search(r"Sex: (?P<sx>.*)$", comment)
        if sx_grp:
            if sx_grp.group("sx").upper().startswith("F"):
                sex = 1.0
            elif sx_grp.group("sx").upper().startswith("M"):
                sex = 0.0
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

                yield f


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


class RangeKeyDict(dict):
    """Custom 'dictionary' for reverse lookup of range
    """
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
                (
                    value
                    for key, value in self.items()
                    if key[0] <= number < key[1]
                )
            )
        except StopIteration:
            raise KeyError(number)
        return result

    def get(self, number, default=None):
        try:
            return self.__getitem__(number)
        except KeyError:
            return default
