#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include "neighbor.cuh"

namespace py = pybind11;


PYBIND11_MODULE(libneighbor, m) {
    py::class_<NeighborList>(m, "NeighborList")
        .def(py::init<int, int, float>())
        .def("update_box", &NeighborList::update_box)
        .def("find_neighbor", &NeighborList::find_neighbor)
        .def("find_cell_list", &NeighborList::find_cell_list)
        .def("convert_ijS", &NeighborList::convert_ijS)
        .def_readonly("N_neighbor", &NeighborList::N_neighbor);
}
