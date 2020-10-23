#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generate PDFs that resemble actual ECG trace printouts (for technician verification)

import argparse
import os
import re
from functools import partial
from multiprocessing import Pool, freeze_support
from xml.etree import ElementTree as ET

from matplotlib import pyplot as plt
from PySierraECG import get_leads
from tqdm import tqdm


def get_tree_val(tree, tree_key, p_key="globalmeasurements", ns={"ns": "http://www3.medical.philips.com"}):
    c = tree.find(f".//ns:{p_key}/ns:{tree_key}", namespaces=ns)
    if c is None:
        return None
    else:
        return c.text


def calculate_tick_range(min_y, max_y, step):
    y_min_offset = min_y % step
    y_max_offset = max_y % step
    return range(int(min_y - y_min_offset), int(max_y - y_max_offset + step), step)


def generate_pdf(xml_fn, out_dir="pdf", tick_aspect=10/4, main_lead_key="II"):
    # check if ecg_id filename is valid
    r = re.match(
        r".*(?P<id>[\w]{8}-[\w]{4}-[\w]{4}-[\w]{4}-[\w]{12})\.xml", xml_fn)
    if not r:
        return None, None
    ecg_id = r.group("id")

    # check if we can get the leads from the filename
    try:
        leads = get_leads(xml_fn)
    except Exception as e:
        return ecg_id, str(e)

    # check if we can load and parse the XML
    try:
        tree = ET.parse(xml_fn)
    except Exception as e:
        return ecg_id, str(e)

    # check durations and number of samples match the data array
    try:
        assert len(leads) == 12, "lead count not equal to 12"
        ref_ms_per_sample = None
        names = ["I", "II", "III", "aVR", "aVL",
                 "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for lead in leads:
            ms_per_sample = lead["duration"] // lead["nsamples"]
            if ref_ms_per_sample is None:
                ref_ms_per_sample = ms_per_sample
            else:
                assert ms_per_sample == ref_ms_per_sample, "ms per sample mismatch"
            x = range(0, lead["duration"], ms_per_sample)
            assert len(x) == lead["nsamples"], "duration nsamples mismatch"
            assert lead["name"] in names, f"invalid lead name: {lead['name']}"
            # assert max(lead['data']) <= 1000, "WARN: lead signal max over 5mV"
            # assert min(lead['data']) >= -1000, "WARN: lead signal min under -5mV"
            names.pop(names.index(lead["name"]))
    except Exception as e:
        return ecg_id, str(e)
    finally:
        del names
        del ref_ms_per_sample

    leads_dict = {}
    while leads:
        lead = leads.pop()
        name = lead.pop("name")
        leads_dict[name] = lead["data"]
    del leads
    del name

    # according to the calibration signal, 0-200 is equal to 1mV
    # we want to make a grid where all of the ticks are in millimeters
    # standard calibration: 10mm/mv (y-axis), 25mm/sec (x-axis)
    # y: 5mm == 100 units; x: 5mm == 40 ms

    # I,   aVR, V1, V4
    # II,  aVL, V2, V5
    # III, aVF, V3, V6
    # II

    q_x = len(x) // 4   # plot column separator width
    q_y = 600           # plot row separator height

    plot_leads = {
        "MAIN": leads_dict[main_lead_key],
        "I":    [v + 3 * q_y for v in leads_dict["I"]][:q_x],
        "II":   [v + 2 * q_y for v in leads_dict["II"]][:q_x],
        "III":  [v + 1 * q_y for v in leads_dict["III"]][:q_x],
        "aVR":  [v + 3 * q_y for v in leads_dict["aVR"]][q_x:2 * q_x],
        "aVL":  [v + 2 * q_y for v in leads_dict["aVL"]][q_x:2 * q_x],
        "aVF":  [v + 1 * q_y for v in leads_dict["aVF"]][q_x:2 * q_x],
        "V1":   [v + 3 * q_y for v in leads_dict["V1"]][2 * q_x:3 * q_x],
        "V2":   [v + 2 * q_y for v in leads_dict["V2"]][2 * q_x:3 * q_x],
        "V3":   [v + 1 * q_y for v in leads_dict["V3"]][2 * q_x:3 * q_x],
        "V4":   [v + 3 * q_y for v in leads_dict["V4"]][3 * q_x:4 * q_x],
        "V5":   [v + 2 * q_y for v in leads_dict["V5"]][3 * q_x:4 * q_x],
        "V6":   [v + 1 * q_y for v in leads_dict["V6"]][3 * q_x:4 * q_x]
    }

    # some of the data has signals that are extreme outliers and overlap over the other leads
    y_min = min([min(l) for l in plot_leads.values()])
    y_max = max([max(l) for l in plot_leads.values()])
    x = range(0, lead["duration"], lead["duration"] // lead["nsamples"])

    # if the scaled ecg signal y-axis is longer than the time x-axis, that's an error (breaks all text)
    if (y_max - y_min) * (tick_aspect) > max(x):
        return ecg_id, "mv signal has greater distribution than allowable ms time range"

    fig, ax = plt.subplots(figsize=(10.5 * 2, 8.0 * 2), dpi=300)
    ax.plot(x, plot_leads["MAIN"])
    ax.set_xticks(range(0, max(x)+1, 40), minor=True)
    ax.set_xticks(range(0, max(x)+1, 200), minor=False)

    # plot columns [I, II, III]
    ax.plot(x[:q_x], plot_leads["I"])
    ax.plot(x[:q_x], plot_leads["II"])
    ax.plot(x[:q_x], plot_leads["III"])
    # plot columns [aVR, aVL, aVF]
    ax.plot(x[q_x:2 * q_x], plot_leads["aVR"])
    ax.plot(x[q_x:2 * q_x], plot_leads["aVL"])
    ax.plot(x[q_x:2 * q_x], plot_leads["aVF"])
    # plot columns [V1, V2, V3]
    ax.plot(x[2 * q_x:3 * q_x], plot_leads["V1"])
    ax.plot(x[2 * q_x:3 * q_x], plot_leads["V2"])
    ax.plot(x[2 * q_x:3 * q_x], plot_leads["V3"])
    # plot columns [V4, V5, V6]
    ax.plot(x[3 * q_x:4 * q_x], plot_leads["V4"])
    ax.plot(x[3 * q_x:4 * q_x], plot_leads["V5"])
    ax.plot(x[3 * q_x:4 * q_x], plot_leads["V6"])

    # set label text for each lead
    q_xd = max(x) // 4
    text_kwargs = {"horizontalalignment": "left",
                   "verticalalignment": "top", "fontsize": 14}
    ax.text(0 * q_xd, 0 * q_y + 200, main_lead_key, **text_kwargs)  # MAIN
    ax.text(0 * q_xd, 3 * q_y + 200, "I", **text_kwargs)
    ax.text(0 * q_xd, 2 * q_y + 200, "II", **text_kwargs)
    ax.text(0 * q_xd, 1 * q_y + 200, "III", **text_kwargs)
    ax.text(1 * q_xd, 3 * q_y + 200, "aVR", **text_kwargs)
    ax.text(1 * q_xd, 2 * q_y + 200, "aVL", **text_kwargs)
    ax.text(1 * q_xd, 1 * q_y + 200, "aVF", **text_kwargs)
    ax.text(2 * q_xd, 3 * q_y + 200, "V1", **text_kwargs)
    ax.text(2 * q_xd, 2 * q_y + 200, "V2", **text_kwargs)
    ax.text(2 * q_xd, 1 * q_y + 200, "V3", **text_kwargs)
    ax.text(3 * q_xd, 3 * q_y + 200, "V4", **text_kwargs)
    ax.text(3 * q_xd, 2 * q_y + 200, "V5", **text_kwargs)
    ax.text(3 * q_xd, 1 * q_y + 200, "V6", **text_kwargs)

    # major y-ticks every 100, minor y-ticks every 20. major ticks aligned to 0
    # ax.set_yticks(range(-100, 250+1, 20), minor=True)
    # ax.set_yticks(range(-100, 250+1, 100), minor=False)
    ax.set_yticks(calculate_tick_range(y_min, y_max, 20), minor=True)
    ax.set_yticks(calculate_tick_range(y_min, y_max, 100), minor=False)
    ax.set_xticklabels(())
    ax.set_yticklabels(())

    # set the gridlines
    ax.grid(b=True, which="major", axis="both",
            color="pink", linestyle="solid", linewidth=1.5)
    ax.grid(b=True, which="minor", axis="both",
            color="pink", linestyle="solid", linewidth=0.5)

    # force aspect ratio to have gridlines square box shaped
    ax.set_aspect(tick_aspect)
    plt.margins(0.01)

    ax.set_xlabel("25 ticks/second")
    ax.set_ylabel("10 ticks/mV")

    # hardcoded text metadata, assumes 11000 millisecond duration for x-axis offset
    x_off = 0
    text_kwargs = {"horizontalalignment": "left",
                   "verticalalignment": "bottom", "fontsize": 10}
    ns = {"ns": "http://www3.medical.philips.com"}

    # construct acquisition & patient information
    ecg_metadata = f"ecgId: {ecg_id}\n"
    ecg_metadata += "\nAcquisition"
    ecg_metadata += "\n  Date:"
    ecg_metadata += "\n  Time:"
    ecg_metadata += "\nPatient ID:"
    ecg_metadata += "\nBirth Date:"
    ecg_metadata += "\nSex:"
    ecg_metadata += "\nHeight:"
    ecg_metadata += "\nWeight:"
    ax.text(x_off, y_max + 50, ecg_metadata, **text_kwargs)
    x_off += 650

    patientid = get_tree_val(tree, "patientid", p_key="generalpatientdata")
    acq_node = tree.find("./ns:dataacquisition", namespaces=ns)
    acq_date = acq_node.attrib.get("date", None)
    acq_time = acq_node.attrib.get("time", None)
    ecg_metadata = f"\n{acq_date}\n{acq_time}"
    ecg_metadata += f"\n{patientid}"
    dateofbirth = get_tree_val(tree, "dateofbirth", p_key="age")
    ecg_metadata += f"\n{dateofbirth}"
    sex = get_tree_val(tree, "sex", p_key="generalpatientdata")
    ecg_metadata += f"\n{sex}"
    height = get_tree_val(tree, "cm", p_key="height")
    ecg_metadata += f"\n{height} cm"
    weight = get_tree_val(tree, "kg", p_key="weight")
    ecg_metadata += f"\n{weight} kg"
    ax.text(x_off, y_max + 50, ecg_metadata, **text_kwargs)
    x_off += 850

    # construct duration
    ecg_metadata = "\nRR Interval:"
    ecg_metadata += "\nP Duration:"
    ecg_metadata += "\nPR Interval:"
    ecg_metadata += "\nQRS Duration:"
    ecg_metadata += "\nQT Interval:"
    ecg_metadata += "\nQTCb:"
    ecg_metadata += "\nQTCf:"
    ecg_metadata += "\nQ Onset:"
    ax.text(x_off, y_max + 50, ecg_metadata, **text_kwargs)
    x_off += 750
    rrint = get_tree_val(tree, "rrint")
    ecg_metadata = f"\n{rrint} ms"
    pdur = get_tree_val(tree, "pdur")
    ecg_metadata += f"\n{pdur} ms"
    pr_int = get_tree_val(tree, "print")
    ecg_metadata += f"\n{pr_int} ms"
    qrs_dur = get_tree_val(tree, "qrsdur")
    ecg_metadata += f"\n{qrs_dur} ms"
    qt_int = get_tree_val(tree, "qtint")
    ecg_metadata += f"\n{qt_int} ms"
    qtcb = get_tree_val(tree, "qtcb")
    ecg_metadata += f"\n{qtcb} ms"
    qtcf = get_tree_val(tree, "qtcf")
    ecg_metadata += f"\n{qtcf} ms"
    qonset = get_tree_val(tree, "qonset")
    ecg_metadata += f"\n{qonset} ms"
    ax.text(x_off, y_max + 50, ecg_metadata, **text_kwargs)
    x_off += 800

    # construct frontaxis
    ecg_metadata = "Heart rate:\n"
    ecg_metadata += "\nP Front Axis:"
    ecg_metadata += "\ni40 Front Axis:"
    ecg_metadata += "\nt40 Front Axis:"
    ecg_metadata += "\nQRS Front Axis:"
    ecg_metadata += "\nST Front Axis:"
    ecg_metadata += "\nT Front Axis:"
    ax.text(x_off, y_max + 50, ecg_metadata, **text_kwargs)
    x_off += 800
    heartrate = get_tree_val(tree, "heartrate")
    ecg_metadata = f"{heartrate} BPM\n"
    pfrontaxis = get_tree_val(tree, "pfrontaxis")
    ecg_metadata += f"\n{pfrontaxis}°"
    i40frontaxis = get_tree_val(tree, "i40frontaxis")
    ecg_metadata += f"\n{i40frontaxis}°"
    t40frontaxis = get_tree_val(tree, "t40frontaxis")
    ecg_metadata += f"\n{i40frontaxis}°"
    qrsfrontaxis = get_tree_val(tree, "qrsfrontaxis")
    ecg_metadata += f"\n{qrsfrontaxis}°"
    stfrontaxis = get_tree_val(tree, "stfrontaxis")
    ecg_metadata += f"\n{stfrontaxis}°"
    tfrontaxis = get_tree_val(tree, "tfrontaxis")
    ecg_metadata += f"\n{i40frontaxis}°"
    ax.text(x_off, y_max + 50, ecg_metadata, **text_kwargs)
    x_off += 800

    # construct horizaxis
    ecg_metadata = "Atrial rate:\n"
    ecg_metadata += "\nP Horiz. Axis:"
    ecg_metadata += "\ni40 Horiz. Axis:"
    ecg_metadata += "\nt40 Horiz. Axis:"
    ecg_metadata += "\nQRS Horiz. Axis:"
    ecg_metadata += "\nST Horiz. Axis:"
    ecg_metadata += "\nT Horiz. Axis:"
    ax.text(x_off, y_max + 50, ecg_metadata, **text_kwargs)
    x_off += 800
    atrialrate = get_tree_val(tree, "atrialrate")
    ecg_metadata = f"{atrialrate} BPM\n"
    phorizaxis = get_tree_val(tree, "phorizaxis")
    ecg_metadata += f"\n{phorizaxis}°"
    i40horizaxis = get_tree_val(tree, "i40horizaxis")
    ecg_metadata += f"\n{i40horizaxis}°"
    t40horizaxis = get_tree_val(tree, "t40horizaxis")
    ecg_metadata += f"\n{i40horizaxis}°"
    qrshorizaxis = get_tree_val(tree, "qrshorizaxis")
    ecg_metadata += f"\n{qrshorizaxis}°"
    sthorizaxis = get_tree_val(tree, "sthorizaxis")
    ecg_metadata += f"\n{sthorizaxis}°"
    thorizaxis = get_tree_val(tree, "thorizaxis")
    ecg_metadata += f"\n{i40horizaxis}°"
    ax.text(x_off, y_max + 50, ecg_metadata, **text_kwargs)
    x_off += 800

    # construct severity and codes
    severity = get_tree_val(tree, "severity", p_key="interpretation")
    ecg_metadata = f"{severity}\n"
    mdsignatureline = get_tree_val(tree, "mdsignatureline", p_key="interpretation")
    ecg_metadata += f"{mdsignatureline}\n"
    codes = zip(
        [x.text or "" for x in tree.findall(
            f".//ns:statement/ns:statementcode", namespaces=ns)],
        [x.text or "" for x in tree.findall(
            f".//ns:statement/ns:leftstatement", namespaces=ns)],
        [x.text or "" for x in tree.findall(
            f".//ns:statement/ns:rightstatement", namespaces=ns)]
    )
    for code in codes:
        ecg_metadata += "\n" + " ".join(code)
    ax.text(x_off, y_max + 50, ecg_metadata, **text_kwargs)

    # plt.show()
    fig_name = os.path.join(out_dir, f"{ecg_id}.pdf")
    fig.savefig(fig_name, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return ecg_id, None


def main():
    try:
        # multiprocessing number of workers
        procs = len(os.sched_getaffinity(0))
    except:
        procs = 1

    p = argparse.ArgumentParser(
        description="Create summary PDFs from Sierra ECG XML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--in-dir",
                   default=os.path.join(os.getcwd(), "out", "full-valid"),
                   help="directory of *.xml files")
    p.add_argument("-o", "--out-dir",
                   default=os.path.join(os.getcwd(), "out", "pdf"),
                   help="pdf output directory")
    p.add_argument("-p", "--pool", default=procs, type=int,
                   help="number of multiprocessing cores to use, set 0 for no pooling")
    args = vars(p.parse_args())

    assert os.path.isdir(args["in_dir"]), f"Missing -i {args['in_dir']}"
    out_dir = os.path.abspath(args["out_dir"])
    print(f"Output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    full_xml_files = [os.path.join(args["in_dir"], p)
                      for p in os.listdir(args["in_dir"])]
    #   if p == "43a2c000-89a6-11e7-4823-0014e8680029.xml"]  # debug
    #   if p == "2f915c02-4536-4b21-a08b-eb2d1ddbc21a.xml"]  # debug
    with tqdm(full_xml_files, desc="Generating PDFs") as t:
        if args["pool"] <= 0:
            out = [generate_pdf(ecg_id, out_dir=out_dir)
                   for ecg_id in t]
        else:
            with Pool(args["pool"], initializer=tqdm.set_lock,
                      initargs=(tqdm.get_lock(),)) as p:
                out = list(p.imap_unordered(
                    partial(generate_pdf, out_dir=out_dir), t))

    invalid = 0
    for ecg_id, errors in sorted(out):
        if errors:
            print(ecg_id, errors)
            invalid += 1
    valid = len(out) - invalid
    print(f"num invalid: {invalid}, num valid: {valid}")


if __name__ == "__main__":
    main()
