import struct
from typing import List
import numpy as np


class VECX_DTYPE:
    FLOAT32 = 1
    QINT8 = 2

    def is_valid(dtype):
        return dtype in [
            VECX_DTYPE.FLOAT32,
            VECX_DTYPE.QINT8,
        ]

    def sizeof(dtype):
        if dtype == VECX_DTYPE.FLOAT32:
            return 4
        if dtype == VECX_DTYPE.QINT8:
            return 1
        raise Exception(f"Unhandled dtype {dtype}")


class Vecx:
    def __init__(
        self,
        data: List[any] | np.ndarray,
        *,
        dtype=VECX_DTYPE.FLOAT32,
        auto_quantize=False,
        symmetric=False,
        scale=None,
        zero=None,
    ):
        self.np_data = (
            np.array(data, dtype=np.float32) if isinstance(data, list) else data
        )
        self.dtype = dtype
        self.scale = scale or 0.0
        self.zero = zero or 0

        if dtype == VECX_DTYPE.FLOAT32:
            if auto_quantize:
                self._determine_scale_and_zero(symmetric or False)
            # try auto-quantizing
            if self.scale < 0.0:
                raise Exception("Quantization scale cannot be 0 or negative")
            elif self.scale > 0.0:
                self._quantize_i8()
        # else user uses the class as a dumb serializer

    # https://huggingface.co/docs/optimum/en/concept_guides/quantization
    def _determine_scale_and_zero(self, symmetric: bool):
        # [min, max] -- fs,z --> [-127, 127]
        qmin, min_val = np.iinfo(np.int8).min, np.min(self.np_data)
        qmax, max_val = np.iinfo(np.int8).max, np.max(self.np_data)
        if symmetric:
            max_abs = max(abs(min_val), abs(max_val))
            min_val = -max_abs
            max_val = max_abs
            self.zero = 0
            self.scale = max_abs / max(abs(qmin), abs(qmax))
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            if scale == 0:
                scale = 1.0
            zero = round(qmin - min_val / scale)
            self.scale = scale
            self.zero = np.clip(zero, qmin, qmax)

    def _quantize_i8(self):
        q = np.round(self.np_data / self.scale + self.zero)
        q = np.clip(q, np.iinfo(np.int8).min, np.iinfo(np.int8).max)
        self.np_data = q.astype(np.int8)
        self.dtype = VECX_DTYPE.QINT8

    def dequantize_to_vec_f32(self):
        if self.dtype != VECX_DTYPE.FLOAT32:
            data = self.scale * (self.np_data.astype(np.float32) - self.zero)
            return Vecx(data, dtype=VECX_DTYPE.FLOAT32)
        return self

    def pack(self) -> bytearray:
        size = self.np_data.size
        blob = bytearray()
        blob += b"vecx"  # 4 bytes
        blob += struct.pack("<B", self.dtype)  # 1 byte
        blob += struct.pack("fi", self.scale, self.zero)  # 4(f32) + 4(i32) = 8 bytes
        blob += struct.pack("<Q", size)  # 8 bytes
        blob += self.np_data.tobytes()
        return blob

    @staticmethod
    def unpack(blob):
        if not isinstance(blob, (bytes, bytearray)):
            raise ValueError("Input must be a bytes-like object")

        if blob[:4] != b"vecx":
            raise ValueError("Invalid magic header")

        dtype = struct.unpack("<B", blob[4:5])[0]

        if not VECX_DTYPE.is_valid(dtype):
            raise ValueError(f"Unsupported dtype: {dtype}")

        scale, zero = struct.unpack("fi", blob[5:13])
        size = struct.unpack("<Q", blob[13:21])[0]

        expected_data_len = size * VECX_DTYPE.sizeof(dtype)
        actual_data = blob[21:]
        if len(actual_data) != expected_data_len:
            raise ValueError(
                f"Data size mismatch: expected {expected_data_len}, got {len(actual_data)}"
            )

        if dtype == VECX_DTYPE.FLOAT32:
            data = np.frombuffer(actual_data, dtype=np.float32).tolist()
        elif dtype == VECX_DTYPE.QINT8:
            data = np.frombuffer(actual_data, dtype=np.int8).tolist()

        return Vecx(data, dtype=dtype, scale=scale, zero=zero)

    def __getitem__(self, index):
        return self.np_data[index]

    def __setitem__(self, _index, _value):
        raise Exception("Cannot set read-only data")

    def __len__(self):
        return self.np_data.size
