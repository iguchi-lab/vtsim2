#include <pybind11/pybind11.h>
#include <stdio.h>

void hello_c(void){
    printf('hello world from C');
}

PYBIND11_MODULE(vtsimc2, m) {
    m.def("hello_c", &hello_c, "");
}