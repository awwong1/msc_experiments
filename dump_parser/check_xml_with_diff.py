#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Checks that the generated xml files are consistent with the SQL insert diff
# Refer to https://archive.org/details/tracemastervueecgmanagementsystemdatabaseandxmlschemadatadictionary

import argparse
import os
import json
from functools import partial
from multiprocessing import Pool, freeze_support
from xml.etree import ElementTree as ET

from tqdm import tqdm


def determine_global_measurements_match(diff_key, tree_key, matches, diff, tree, ns={"ns": "http://www3.medical.philips.com"}):
    a = diff.get(diff_key, None)
    b = tree.find(f".//ns:globalmeasurements/ns:{tree_key}", namespaces=ns)
    if b is None and a is not None:
        matches[diff_key] = f"{a} != {b}"
    elif a is None and (b is None or b.text == "Invalid" or b.text == "Failed"):
        return
    elif str(a) == b.text or a == b.text:
        return
    else:
        matches[diff_key] = f"{a} != {b.text}"


def verify_xml_and_diff(xml_fn, out_dir="out"):
    ecg_id = os.path.splitext(xml_fn)[0]
    # load XML
    pre_check_fp = os.path.join(out_dir, "full", xml_fn)
    tree = ET.parse(pre_check_fp)
    # load JSON diff
    with open(os.path.join(out_dir, "json", f"{ecg_id}.json"), "r") as f:
        diff = json.load(f)

    matches = {}

    # tblECGMain, globalmeasurements
    # heartRate: Heart rate at time of ECG acquisition
    determine_global_measurements_match("heartRate", "heartrate", matches, diff, tree)
    # prInterval: PR Interval
    determine_global_measurements_match("prInterval", "print", matches, diff, tree)
    # qrsDuration: QRS duration
    determine_global_measurements_match("qrsDuration", "qrsdur", matches, diff, tree)
    # qtInterval: QT interval
    determine_global_measurements_match("qtInterval", "qtint", matches, diff, tree)
    # qtcb: QTCB
    determine_global_measurements_match("qtcb", "qtcb", matches, diff, tree)
    # pFrontAxis: P front axis
    determine_global_measurements_match("pFrontAxis", "pfrontaxis", matches, diff, tree)
    # i40FrontAxis: i40 front axis
    determine_global_measurements_match("i40FrontAxis", "i40frontaxis", matches, diff, tree)
    # t40FrontAxis: T40 front axis
    determine_global_measurements_match("t40FrontAxis", "t40frontaxis", matches, diff, tree)
    # qrsFrontAxis: QRS front axis
    determine_global_measurements_match("qrsFrontAxis", "qrsfrontaxis", matches, diff, tree)
    # stFrontAxis: ST front axis
    determine_global_measurements_match("stFrontAxis", "stfrontaxis", matches, diff, tree)
    # tFrontAxis: T front axis
    determine_global_measurements_match("tFrontAxis", "tfrontaxis", matches, diff, tree)
    # pHorizAxis: P horizontal axis
    determine_global_measurements_match("pHorizAxis", "phorizaxis", matches, diff, tree)
    # i40HorizAxis: I40 horizontal axis
    determine_global_measurements_match("i40HorizAxis", "i40horizaxis", matches, diff, tree)
    # t40HorizAxis: T40 horizontal axis
    determine_global_measurements_match("t40HorizAxis", "t40horizaxis", matches, diff, tree)
    # qrsHorizAxis: QRS horizontal axis
    determine_global_measurements_match("qrsHorizAxis", "qrshorizaxis", matches, diff, tree)
    # stHorizAxis: ST horizontal axis
    determine_global_measurements_match("stHorizAxis", "sthorizaxis", matches, diff, tree)
    # tHorizAxis: T horizontal axis
    determine_global_measurements_match("tHorizAxis", "thorizaxis", matches, diff, tree)
    # rrInterval: RR interval
    determine_global_measurements_match("rrInterval", "rrint", matches, diff, tree)
    # pDuration: P duration
    determine_global_measurements_match("pDuration", "pdur", matches, diff, tree)
    # qOnset: Q onset
    determine_global_measurements_match("qOnset", "qonset", matches, diff, tree)
    # tOnset: T onset
    # determine_global_measurements_match("tOnset", "tonset", matches, diff, tree)
    # qtcf: Qtcf
    determine_global_measurements_match("qtcf", "qtcf", matches, diff, tree)
    # qtco: Qtco
    determine_global_measurements_match("qtco", "qtco", matches, diff, tree)

    # make symlinks for valid, invalid
    if len(matches):
        sym = os.path.join(out_dir, "full-invalid", xml_fn)
    else:
        sym = os.path.join(out_dir, "full-valid", xml_fn)

    os.symlink(os.path.relpath(pre_check_fp, start=os.path.join(out_dir, "full-valid")), sym)

    return (ecg_id, matches)


def main():
    try:
        # multiprocessing number of workers
        procs = len(os.sched_getaffinity(0))
    except:
        procs = 1

    p = argparse.ArgumentParser(
        description="Verify Philips SQL DB dump diffs with extracted XML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-o", "--out-dir", default=os.path.join(os.getcwd(),
                                                           "out"), help="sql_decoder.py output directory")
    p.add_argument("-p", "--pool", default=procs, type=int,
                   help="number of multiprocessing cores to use, set 0 for no pooling")
    args = vars(p.parse_args())

    out_dir = os.path.abspath(args["out_dir"])
    print(f"Output: {out_dir}")
    diff_dir = os.path.join(out_dir, "json")
    full_dir = os.path.join(out_dir, "full")
    assert os.path.isdir(diff_dir), f"Missing {diff_dir}"
    assert os.path.isdir(full_dir), f"Missing {full_dir}"

    os.makedirs(os.path.join(out_dir, "full-valid"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "full-invalid"), exist_ok=True)

    full_xml_files = os.listdir(full_dir)
    with tqdm(full_xml_files, desc="Verifying") as t:
        if args["pool"] <= 0:
            out = [verify_xml_and_diff(ecg_id, out_dir=out_dir)
                   for ecg_id in t]
        else:
            with Pool(args["pool"], initializer=tqdm.set_lock,
                      initargs=(tqdm.get_lock(),)) as p:
                out = list(p.imap_unordered(
                    partial(verify_xml_and_diff, out_dir=out_dir), t))

    invalid = 0
    for ecg_id, matches in sorted(out):
        if matches:
            print(ecg_id, matches)
            invalid += 1
    valid = len(out) - invalid
    print(f"num invalid: {invalid}, num valid: {valid}")

if __name__ == "__main__":
    main()
