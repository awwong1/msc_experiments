#include <Python.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> /** to check if a file exist**/
#include "libsierraecg/sierraecg.h"

static PyObject *Error;

typedef struct
{
    PyObject_HEAD
} ecgLead;

static PyTypeObject ecgLeadType = {
    PyVarObject_HEAD_INIT(NULL, 0) "PySierraECG.lead",
    sizeof(ecgLead),
    0,
    Py_TPFLAGS_DEFAULT,
    "Lead voltage and time data of ECG"};

char *concat(const char *s1, const char *s2)
{
    char *result = malloc(strlen(s1) + strlen(s2) + 1);
    if (result == NULL) {
        return NULL;
    }
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

PyObject *getList(size_t length, short *array)
{
    PyObject *list = PyList_New(length);
    for (int i = 0; i < length; ++i)
    {
        // The units are scaled such that 200 == 1mv
        PyList_SetItem(list, i, PyFloat_FromDouble(array[i]));
    }

    return list;
}

PyObject *getData(size_t nleads, lead_t *leads)
{
    PyObject *result = PyList_New(nleads);

    for (int i = 0; i < nleads; ++i)
    {
        PyObject *lead = Py_BuildValue(
            "{s:s, s:i, s:i, s:O}",
            "name", leads[i].name,
            "nsamples", leads[i].count,
            "duration", leads[i].duration,
            "data", getList(leads[i].count, leads[i].samples));

        PyList_SetItem(result, i, lead);
    }

    return result;
}

int checkFile(const char *path)
{
    if (access(path, F_OK) != -1)
    {
        return 1;
    }
    else
    {
        char *msg = concat("File doesn't exist: ", path);

        PyErr_SetString(Error, msg);
        free(msg);
        return 0;
    }
}

static PyObject *
PySierraECG_getLeads(PyObject *self, PyObject *args)
{
    const char *command; /** python arguments transformed to C string **/
    ecg_t ecg;           /** ecg variable with lead data **/
    PyObject *ECG;       /** object result **/

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    if (!checkFile(command))
        return NULL;

    /* Allow the library to initialize */
    if (sierraecg_init())
        abort();

    /* sierraecg_read(input, ecg data) */
    if (sierraecg_read(command, &ecg))
    {
        char *msg = concat("Errors parsing Sierra ECG file: ", command);

        PyErr_SetString(Error, msg);
        free(msg);
    }
    else
    {
        /*
        * ecg.version(char[8]):  Version, i.e. 1.03 or 1.04
        * ecg.leads(lead_t[16]): Retrieved lead data
        * ecg.valid(size_t): How many valid leads were read
        */
        /*
        * lead.name(char*): e.g. I, II, III
        * lead.samples(short[]): Lead sample data
        * lead.count(size_t): Number of samples
        * lead.duration(size_t): Total msec of the recording
        */

        /* get lead data */
        ECG = getData(ecg.valid, ecg.leads);

        /* cleanup the ECG itself */
        sierraecg_free(&ecg);
    }

    /* Allow the library to cleanup its dependencies */
    sierraecg_cleanup();

    return ECG;
}

static PyMethodDef PySierraECGMethods[] = {
    {"get_leads", PySierraECG_getLeads, METH_VARARGS,
     "Get leads data from ECG results in XML Sierra Philips format"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef PySierraECGmodule = {
    PyModuleDef_HEAD_INIT,
    "PySierraECG", /* name of module */
    NULL,          /* module documentation*/
    -1,            /* size of per-interpreter state of the module,
        or -1 if the module keeps state in global variables. */
    PySierraECGMethods};

PyMODINIT_FUNC PyInit_PySierraECG(void)
{
    /** initialize the custom data types **/
    ecgLeadType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ecgLeadType) < 0)
        return NULL;

    /** initialize the module **/
    PyObject *module;
    module = PyModule_Create(&PySierraECGmodule);

    if (module == NULL)
        return NULL;

    /** initialize the custom error variable **/
    Error = PyErr_NewException("PySierraECG.error", NULL, NULL);

    /** add objects to module **/
    Py_INCREF(&ecgLeadType);
    Py_INCREF(Error);
    PyModule_AddObject(module, "lead", (PyObject *)&ecgLeadType);
    PyModule_AddObject(module, "error", Error);

    return module;
}