#include <pybind11/pybind11.h>
#include "popsift.h"

namespace py = pybind11;

PYBIND11_MODULE(pypopsift, m) {
    m.doc() = R"pbdoc(
        pypopsift: Python module for CUDA accelerated GPU SIFT
    )pbdoc";

    m.def("popsift", pps::popsift,
        "Compute SIFT keypoints and descriptors on GPU\n\n"
        "Returns:\n"
        "    [keypoints, descriptors]\n"
        "    keypoints: Shape[n, 4] where the 2nd dimension is [x, y, size, angle]\n"
        "    descriptors: Shape[n, 128]",
        py::arg("image"),
        py::arg("peak_threshold") = 0.04,
        py::arg("edge_threshold") = 10,
        py::arg("use_root") = true,
        py::arg("downsampling") = -1
    );

    m.def("fits_texture", pps::fitsTexture,
        "Check if the image fits in the texture memory",
        py::arg("width"),
        py::arg("height"),
        py::arg("downsampling") = -1
    );

    m.def("cuda_is_available", pps::cudaIsAvailable);

    m.def("get_cuda_memory_info", pps::getCudaMemoryInfo,
        "Get the free and total memory of the current CUDA device\n\n"
        "Returns:\n"
        "    (free_bytes, total_bytes)"
    );

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
