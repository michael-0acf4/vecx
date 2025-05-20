import numpy as np
from vecx_spec import Vecx

x = Vecx([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], auto_quantize=True)

assert np.allclose(x.np_data, np.array([-100, -71, -43, -15, 14, 42, 70, 99, 127, 127]))

# optimal scale and zero offset
assert (x.scale - 0.03529411764705882) < 10e-6
assert (x.zero - (-128)) < 10e-6

assert np.allclose(
    x.dequantize_to_vec_f32().np_data,
    np.array(
        [
            0.98823535,
            2.0117648,
            3.0000002,
            3.9882355,
            5.011765,
            6.0000005,
            6.9882355,
            8.0117655,
            9.0,
            9.0,
        ]
    ),
)
