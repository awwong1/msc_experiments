#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified script to parse Philips TraceMasterVue ECG SQL DB dumps to XML files
# Refer to https://archive.org/details/tracemastervueecgmanagementsystemdatabaseandxmlschemadatadictionary
import io


DB_COLS = (
    "ecgId",
    "dateAcquired",
    "dateReceived",
    "dateConfirmed",
    "UserField4",
    "severityId",
    "headerInfo",
    "userDefines",
    "orderInfo",
    "documentInfo",
    "reportInfo",
    "acquisitionInfo",
    "interpretationInfo",
    "age",
    "ageUnits",
    "Sex",
    "waveformInfo",
    "measurementInfo"
)


def convert_line_to_payload(raw_line):
    line = raw_line.rstrip()
    raw_insert, raw_values = line.split(" VALUES ")
    raw_columns = raw_insert[len("INSERT [dbo].[ECG01] ("):len(raw_insert) - len(")")]  # raw column names
    raw_values = (raw_values[1:len(raw_values) - 1])
    
    raw_columns = raw_columns.split(", ")
    columns = tuple(col[1:len(col) - 1] for col in raw_columns)  # remove surrounding square braces `[`, `]`
    values = raw_values.split(", ")  # , raw_values.split(", "))

    assert columns == DB_COLS, "col mismatch"
    assert len(values) == len(columns), "not enough values for columns"

    return dict(zip(columns, values))


def main():
    with io.open("ecg_sample/dbo.ECG01.Table.sql", encoding="utf-16") as f:
        lines = f.readlines()
    print("".join(lines[:28]))  # anything more is an error

    inserts = lines[28:]
    for ins_idx, raw_insert in enumerate(inserts):
        try:
            payload = convert_line_to_payload(raw_insert)
        except Exception:
            raise Exception(ins_idx)
        print(payload["UserField4"])
        # break



if __name__ == "__main__":
    main()
