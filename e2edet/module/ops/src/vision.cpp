#include <torch/extension.h>
#include "box_attn/box_attn.h"
#include "instance_attn/instance_attn.h"
#include "fast_instance_attn/fast_instance_attn.h"
#include "fast_box_attn/fast_box_attn.h"

namespace e2edet {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("box_attn_forward", &box_attn_forward, "box_attn_forward");
    m.def("box_attn_backward", &box_attn_backward, "box_attn_backward");
    m.def("instance_attn_forward", &instance_attn_forward, "instance_attn_forward");
    m.def("instance_attn_backward", &instance_attn_backward, "instance_attn_backward");
    m.def("fast_instance_attn_forward", &fast_instance_attn_forward, "fast_instance_attn_forward");
    m.def("fast_instance_attn_backward", &fast_instance_attn_backward, "fast_instance_attn_backward");
    m.def("fast_box_attn_forward", &fast_box_attn_forward, "fast_box_attn_forward");
    m.def("fast_box_attn_backward", &fast_box_attn_backward, "fast_box_attn_backward");
}

}