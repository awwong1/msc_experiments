# ecgml_research

Research code for ECG machine learning exploration.
See [Trace Master Vue ECG Management System Database And XML Schema Data Dictionary](https://archive.org/details/tracemastervueecgmanagementsystemdatabaseandxmlschemadatadictionary) for Philips ECG XML and Database Schema definitions.


## Quickstart

```bash
python3 sql_decoder.py input_directory
python3 check_xml_with_diff.py
python3 gen_ecg_pdfs.py
```

## Out Folder

`fragments`
: Raw XML decoded values from `tblECGMain` {headerInfo, userDefines, orderInfo, documentInfo, reportInfo, acquisitionInfo, patientInfo, interpretationInfo} and `tblECGWaveforms` {waveformInfo, measurementInfo}. Column name to decoded value mismatches in [sql_decoder.py.log](./sql_decoder.py.log).

`full`
: XML decoded fragments, joined together based on schema definition. Because column name value mismatches exist, this approach relies on content structure over column names. Warnings in [sql_decoder.py.log](./sql_decoder.py.log).

`full-invalid`
: Symlinks to `full`. These files have data that do not match the reference columns in the `tblECGMain` database table. (example, `tblECGMain` says RR interval should be 1000, but XML file shows 847.) Warnings in [check_xml_with_diff.py.log](./check_xml_with_diff.py.log)

`full-valid`
: Symlinks to `full`. These files have data that correctly match the reference columns in the `tblECGMain` database table.

`json`
: Remaining columns in the `tblECGMain` database that are unused in the XML file. Used for XML to table reference check. Holds additional information (uuids) that are currently unused.

`pdf`
: Technician and cardiologist readable ECG printouts. Errors/warnings shown in [gen_ecg_pdfs.py.log](./gen_ecg_pdfs.py.log)

## Potential Research Questions

### Fri, Jan 24, 2020

* Who has afib diagnosed vs not diagnosed (+/-)
    * these labels are provided, diagnosed by physicians
* Who has stroke vs who does not has a stroke (+/-)
    * these labels are provided, diagnosed by physicians

Can we detect slient afib using historical ECG data from stroke patients? {afib: -, stroke: +}
    * is stroke + and afib = silent afib? (maybe, not always)
    * limit to some time between afib diagnosis and stroke

1. Predict afib given ECG waveform (labels derived from ECG machine itself; investigate how ECG machine does this)

### Wed, Feb 5, 2020

* Outputted PDFs need to be 'calibrated' such that they are readable by ECG technicians & cardiologists
* todo: rework sql extraction, verification, and pdf generation scripts
* Investigate QRS score (manual generated feature)

1. Can we train a model to predict sudden cardiac death probability in ST elevation Myocardial Infarction (STEMI) population?
2. Can we train a model to detect stroke probability in afib population?
