import os
import re
from typing import Iterable, List, Tuple, Union

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


class RangeKeyDict:
    """Custom 'dictionary' for reverse lookup of range
    """
    def __init__(self, my_dict):
        # !any(!A or !B) is faster than all(A and B)
        assert not any(
            map(
                lambda x: not isinstance(x, tuple) or len(x) != 2 or x[0] > x[1],
                my_dict,
            )
        )

        def lte(bound):
            return lambda x: bound <= x

        def gt(bound):
            return lambda x: x < bound

        # generate the inner dict with tuple key like (lambda x: 0 <= x, lambda x: x < 100)
        self._my_dict = {(lte(k[0]), gt(k[1])): v for k, v in my_dict.items()}

    def __getitem__(self, number):
        from functools import reduce

        _my_dict = self._my_dict
        try:
            result = next(
                (
                    _my_dict[key]
                    for key in _my_dict
                    if list(reduce(lambda s, f: filter(f, s), key, [number]))
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
