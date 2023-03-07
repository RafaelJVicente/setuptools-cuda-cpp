#include <Python.h>
#include <cstdio>
#include <string>
#include "cudaext.cuh"

static PyObject *hello_wrapper(PyObject *self, PyObject *args) {
    char *result = hello();
    PyObject *ret = PyBytes_FromString(result);
    return ret;
}

//static PyObject *return_nothing_wrapper(PyObject *self, PyObject *args) {
//    proxylib_init();
//
//    Py_RETURN_NONE;
//}
//
//static PyObject *return_by_value_wrapper(PyObject *self, PyObject *args) {
//    PyObject *obj;
//    if (!PyArg_UnpackTuple(args, "obj", 0, 1, &obj))
//        return NULL;
//    void *input_ptr = PyCapsule_GetPointer(obj, "obj");
//    void *output_Ptr = return_by_value(input_Ptr);
//    PyObject *output = PyCapsule_New(output_Ptr, "outPtr");
//
//    return output;
//}
//
//static PyObject *return_by_reference_wrapper(PyObject *self, PyObject *args) {
//    PyObject *obj;
//    if (!PyArg_UnpackTuple(args, "obj", 0, 1, &obj))
//        return NULL;
//    char *buffer;
//    int length;
//    return_by_reference(PyCapsule_GetPointer(obj, "obj"), &buffer, &length);
//    return PyByteArray_FromStringAndSize(buffer, length);
//}
//
//static PyObject *return_by_value_and_by_reference_wrapper(PyObject *self, PyObject *args) {
//    PyObject *obj;
//    if (!PyArg_UnpackTuple(args, "obj", 0, 1, &obj))
//        return NULL;
//    char *buffer;
//    int length;
//    void *ptr = return_by_value_and_by_reference(PyCapsule_GetPointer(obj, "obj"), &buffer, &length);
//    PyObject *capsule_Ptr = PyCapsule_New(ptr, "ptr");
//    PyObject *py_buffer = PyByteArray_FromStringAndSize(buffer, length);
//    PyObject *tuple_Ptr = PyTuple_Pack(2, capsule_Ptr, py_buffer);
//    return tuple_Ptr;
//}

static PyMethodDef CudaextMethods[] = {
        {"hello", hello_wrapper, METH_NOARGS},
//        {"return_nothing",                   return_nothing_wrapper,                     METH_NOARGS},
//        {"return_by_value",                  return_by_value_wrapper,                    METH_VARARGS},
//        {"return_by_reference",              "return_by_reference_wrapper",              METH_VARARGS},
//        {"return_by_value_and_by_reference", "return_by_value_and_by_reference_wrapper", METH_VARARGS},
        {nullptr,    nullptr,          0, nullptr}
};

// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for it's method definitions
static struct PyModuleDef cudaext_definition = {
        PyModuleDef_HEAD_INIT,
        "cudaext",
        "A Python module extension for C++ lib",
        -1,
        CudaextMethods
};

PyMODINIT_FUNC PyInit_cudaext(void) {
    return PyModule_Create(&cudaext_definition);
}
