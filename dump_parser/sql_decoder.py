#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Decodes Philips TraceMasterVue ECG SQL DB dumps to XML files
# Refer to https://archive.org/details/tracemastervueecgmanagementsystemdatabaseandxmlschemadatadictionary

import argparse
import io
import json
import os
import re
import subprocess
import zlib
from csv import reader
from functools import partial
from multiprocessing import Pool, freeze_support
from xml.dom.minidom import parseString

from tqdm import tqdm

ECG_MAIN_COLUMNS = ("ecgId", "locationId", "patientId", "dateOfBirth", "age", "ageUnits", "height", "heightUnits", "weight", "weightUnits", "sex", "dateAcquired", "dateReceived", "dateModified", "dateConfirmed", "inboxId", "typeId", "sourceId", "severityId", "statFlag", "state", "requestingMD", "mdSignature", "confirmingUserId", "operatorId", "criteriaVersion", "measurementVersion", "heartRate", "prInterval", "qrsDuration", "qtInterval", "qtcb", "pFrontAxis", "i40FrontAxis", "t40FrontAxis", "qrsFrontAxis", "stFrontAxis", "tFrontAxis", "pHorizAxis", "i40HorizAxis", "t40HorizAxis", "qrsHorizAxis", "stHorizAxis", "tHorizAxis",
                    "comparisonEcgId", "headerInfo", "userDefines", "orderInfo", "documentInfo", "reportInfo", "acquisitionInfo", "patientInfo", "interpretationInfo", "ecgTimestamp", "userField1", "userField2", "userField3", "userField4", "userField5", "userField6", "userField7", "userField8", "rrInterval", "pDuration", "qOnset", "tOnset", "qtcf", "qtco", "tOffsetStabilityRank", "ecgFlags", "uniqueOrderId", "orderNumber", "dateOrderRequested", "dateOrderREconciled", "dateAcquiredOffset", "dateAcquiredDST", "dateConfirmedOffset", "dateConfirmedDST", "dateReceivedOffset", "dateReceivedDST", "dateModifiedOffset", "dateModifiedDST")
ECG_WAVEFORMS_COLUMNS = ("ecgId", "waveformInfo",
                         "measurementInfo", "dateAcquired")


def join_ecg_fragments(ecg_id, out_dir="out"):
    """Attempt to join the xml files together, ignoring invalid XML column matches
    """
    ecg_frags_dir = os.path.join(out_dir, "fragments", ecg_id)
    ecg_frags = os.listdir(ecg_frags_dir)
    frag_contents = []
    for ecg_frag in ecg_frags:
        with open(os.path.join(ecg_frags_dir, ecg_frag), "r") as fg:
            frag_contents.append(fg.read())

    # All joined XML files must have header information
    header_idx = None
    for f_idx, content in enumerate(frag_contents):
        if content.startswith("<restingecgdata"):
            if header_idx is not None:
                return ecg_id, ("duplicate header data found",)
            header_idx = f_idx
    if header_idx is None:
        return ecg_id, ("no header data found",)
    xml_part = frag_contents.pop(header_idx)

    # All remaining XML fragments must be parseable and uniquely tagged
    tag_content_map = {}
    while frag_contents:
        frag_content = frag_contents.pop()
        try:
            dom = parseString(frag_content)
        except Exception as e:
            return ecg_id, ("xml frag parse error", str(e))
        tag = dom.documentElement.tagName
        if tag in tag_content_map:
            return ecg_id, ("duplicate tag found", tag)
        tag_content_map[tag] = frag_content

    # all XML files must have a waveform representation
    if "waveforms" not in tag_content_map:
        return ecg_id, ("no waveforms tag found",)

    # reconstruct the full XML file
    for k in sorted(list(tag_content_map.keys())):
        xml_part += tag_content_map[k]
    xml_part += "</restingecgdata>"

    # save the XML file prettily
    try:
        dom = parseString(xml_part)
        with open(os.path.join(out_dir, "full", f"{ecg_id}.xml"), "w") as f:
            dom.writexml(f, addindent="\t", newl="\n")
    except Exception as e:
        return ecg_id, ("xml parse/save error", str(e),)
    return ecg_id, ()


def print_parse_statuses(out_statuses, prefix="status"):
    invalid = [(k, v) for (k, v) in out_statuses if k !=
               None and v != None and len(v) > 0]
    for (k, v) in sorted(invalid):
        print(f"{prefix}: {k} WARN {v}")
    print(f"{prefix}: num invalid/mismatched", len(invalid))
    valid = [(k, v) for (k, v) in out_statuses if v != None and len(v) == 0]
    print(f"{prefix}: num valid", len(valid))


def xxd_to_zlib_flate(raw_data_str):
    """
    convert the sql hexadecimal values into raw xml representation
    """
    # assert raw_data_str.startswith("0x")
    try:
        # it is a well formed hexadecimal zlib binary
        zlib_out = zlib.decompress(bytes.fromhex(raw_data_str[2:]))
    except Exception as e:
        # fallback to subprocess and zlib-flate, truncate errors
        xxd = subprocess.Popen(("xxd", "-r", "-p"), stdin=subprocess.PIPE,
                               stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        xxd_out = xxd.communicate(input=raw_data_str.encode())[0]
        zlib_flate = subprocess.Popen(
            ("zlib-flate", "-uncompress"),
            stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        zlib_out = zlib_flate.communicate(input=xxd_out)[0]
    finally:
        return zlib_out


def check_key_content(k, content):
    """Provided db column name does not always match XML decoded content"""
    if k == "headerInfo" and content.startswith("<restingecgdata"):
        return True
    elif k == "userDefines" and content.startswith("<userdefines"):
        return True
    elif k == "documentInfo" and content.startswith("<documentinfo"):
        return True
    elif k == "reportInfo" and content.startswith("<reportinfo"):
        return True
    elif k == "acquisitionInfo" and content.startswith("<dataacquisition"):
        return True
    elif k == "patientInfo" and content.startswith("<patient"):
        return True
    elif k == "interpretationInfo" and content.startswith("<interpretations"):
        return True
    elif k == "waveformInfo" and content.startswith("<waveforms"):
        return True
    elif k == "measurementInfo" and content.startswith("<internalmeasurements"):
        return True
    elif k == "orderInfo" and content.startswith("<orderinfo"):
        return True
    else:
        # Column name does not match content, or does not exist
        return False


def count_lines(fp, encoding):
    with io.open(fp, "r", encoding=encoding) as f:
        for i, l in enumerate(f):
            pass
    return i+1


def parse_ecg_row(line, out_dir="out", inp_enc="utf-16"):
    """Parse sql file INSERT statements. Returns a tuple (id, status)
    """
    sql_insert = get_sql_insert_as_dict(line)
    ecg_id = sql_insert.get("ecgId")
    if not ecg_id:
        return None, None

    is_ecg_main = any(key in ("headerInfo", "userDefines", "orderInfo", "documentInfo", "reportInfo",
                              "acquisitionInfo", "patientInfo", "interpretationInfo",) for key in sql_insert.keys())
    if not is_ecg_main:
        is_waveforms = list(sql_insert.keys()) == ECG_WAVEFORMS_COLUMNS

    id_dir = os.path.join(out_dir, "fragments", ecg_id)
    os.makedirs(id_dir, exist_ok=True)

    invalid_keys = []
    used_keys = []
    for k, v in sql_insert.items():
        if v and type(v) == str and re.match(r"^0[xX][0-9a-fA-F]+", v):
            # hexadecimal value, zlib_decompress it
            inlated_bytes = xxd_to_zlib_flate(v)

            # EDGE CASE in SQL FILE
            contents = None
            err = None
            try:
                contents = inlated_bytes.decode("utf-16")
            except Exception as e:
                err = e
            try:
                possible_contents = inlated_bytes.decode("utf-8")
                if "\x00" not in possible_contents:
                    contents = possible_contents
            except Exception as e:
                err = e

            if not contents:
                contents = str(err)

            fp = os.path.join(id_dir, f"{k}.xml")
            if not check_key_content(k, contents):
                fp = os.path.join(id_dir, f"{k}.invalid.xml")
                invalid_keys.append(k)
            with open(fp, "w") as f:
                f.write(contents)
            used_keys.append(k)
    if is_ecg_main:
        for k in used_keys:
            sql_insert.pop(k, None)
        with open(os.path.join(out_dir, "json", f"{ecg_id}.json"), "w") as f:
            json.dump(sql_insert, f)
    return ecg_id, invalid_keys


def get_sql_insert_as_dict(line):
    """Given some poorly formatted input, loosely generate a corresponding key/value dictionary
    """
    cols = []
    vals = []

    if line.startswith("INSERT ") and ") VALUES (" in line:
        # the line is a SQL insert line
        raw_columns, raw_values = line.split(" VALUES ")

        # format the raw_columns and raw_values into zipable lists
        col_m = re.match(r"^INSERT.*\((?P<colnames>.*)\)$", raw_columns)
        if col_m:
            # convert "[foo], [bar]" into ["foo", "bar"]
            cols = [colname[1:-1]
                    for colname in col_m.group("colnames").split(", ")]
        val_m = re.match(r"^\((?P<values>.*)\)\n$", raw_values)
        if val_m:
            vals = val_m.group("values").split(", ")
            # quick and dirty value conversion
            for idx, value in enumerate(vals):
                if re.match(r"^N'[A-Za-z0-9-]*'$", value):
                    vals[idx] = value[2:-1]
                elif re.match(r"^([+-]?[0-9]\d*|0)$", value):
                    vals[idx] = int(value)
                elif value == "NULL":
                    vals[idx] = None
    else:
        # the line is a potential comma separated value
        try:
            f = io.StringIO(line)
            csv_reader = reader(f, delimiter=",")
            csv_contents = [r for r in csv_reader]
            assert len(csv_contents) == 1, "only one CSV line should exist"
            vals = csv_contents[0]

            if not any(val.startswith("0x") for val in vals):
                # no hexadecimal string in the data, must not be something we care about
                vals = []
            else:
                # we don't know what the headings are, but the first column is ecgid
                cols = ["ecgId",] + ["Unknown" + str(c) for c in range(1, len(vals))]

            # quick and dirty value conversion (for CSVs)
            for idx, value in enumerate(vals):
                if re.match(r"^([+-]?[0-9]\d*|0)$", value):
                    vals[idx] = int(value)
                elif value == "NULL":
                    vals[idx] = None
        except Exception:
            pass

    return dict(zip(cols, vals))


def main():
    req_files = ["dbo.tblECGMain.Table.sql", "dbo.tblECGWaveforms.Table.sql"]
    try:
        # multiprocessing number of workers
        procs = len(os.sched_getaffinity(0))
    except:
        procs = 1

    p = argparse.ArgumentParser(
        description="Parse Philips SQL DB dumps to XML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "in-dir", help=f"directory containing: ECGMain and ECGWaveforms tables")
    p.add_argument("-o", "--out-dir", default=os.path.join(os.getcwd(),
                                                           "out"), help="xml output directory")
    p.add_argument("-p", "--pool", default=procs, type=int,
                   help="number of multiprocessing cores to use, set 0 for no pooling")
    p.add_argument("-e", "--encoding", default="utf-16",
                   help="input file encoding")
    args = vars(p.parse_args())

    out_dir = os.path.abspath(args["out_dir"])
    print(f"Output: {out_dir}")

    # check file existence
    frag_dir = os.path.join(out_dir, "fragments")
    os.makedirs(frag_dir, exist_ok=True)
    diff_dir = os.path.join(out_dir, "json")
    os.makedirs(diff_dir, exist_ok=True)
    full_dir = os.path.join(out_dir, "full")
    os.makedirs(full_dir, exist_ok=True)

    # get file line count, then parse the file
    print("Parsing SQL files...")
    for req_file in os.listdir(args["in-dir"]):
        fp = os.path.join(args["in-dir"], req_file)
        # skip if file does not end in .sql or .csv
        if not os.path.isfile(fp) and not (fp.endswith(".sql") or fp.endswith(".csv")):
            continue

        total = count_lines(fp, args["encoding"])
        with io.open(fp, "r", encoding=args["encoding"]) as f:
            with tqdm(f, desc=fp, total=total) as t:
                if args["pool"] <= 0:
                    out = [parse_ecg_row(line, out_dir=out_dir, inp_enc=args["encoding"])
                           for line in t]
                else:
                    with Pool(args["pool"], initializer=tqdm.set_lock,
                              initargs=(tqdm.get_lock(),)) as p:
                        out = list(p.imap_unordered(
                            partial(parse_ecg_row, out_dir=out_dir, inp_enc=args["encoding"]), t))
        print_parse_statuses(out, prefix=req_file)

    # combine the XML fragments into one xml file
    print("Joining XML fragments...")
    ecg_ids = [f for f in os.listdir(
        frag_dir) if os.path.isdir(os.path.join(frag_dir, f))]
    with tqdm(ecg_ids, desc="Joining fragments") as t:
        if args["pool"] <= 0:
            out = [join_ecg_fragments(ecg_id, out_dir=out_dir) for ecg_id in t]
        else:
            with Pool(args["pool"], initializer=tqdm.set_lock,
                      initargs=(tqdm.get_lock(),)) as p:
                out = list(p.imap_unordered(
                    partial(join_ecg_fragments, out_dir=out_dir), t))
    print_parse_statuses(out, prefix="Join")


if __name__ == "__main__":
    freeze_support()  # windows support
    main()
