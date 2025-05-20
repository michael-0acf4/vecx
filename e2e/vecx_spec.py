import struct
import numpy as np


class VECX_DTYPE:
    INT32 = 1
    FLOAT32 = 2
    QINT8 = 3
    QUINT8 = 4


def pack_vecx_f32_blob(floats):
    size = len(floats)
    blob = bytearray()
    blob += b"vecx"  # 4 bytes
    blob += struct.pack("<B", VECX_DTYPE.FLOAT32)  # 1 byte
    blob += struct.pack("<Q", size)  # 8 bytes
    blob += np.array(floats, dtype=np.float32).tobytes()
    return blob


def unpack_vecx_f32_blob(blob: bytes):
    if not isinstance(blob, (bytes, bytearray)):
        raise ValueError("Input must be a bytes-like object")

    if blob[:4] != b"vecx":
        raise ValueError("Invalid magic header")

    dtype = struct.unpack("<B", blob[4:5])[0]
    if dtype != VECX_DTYPE.FLOAT32:
        raise ValueError(f"Unsupported dtype: {dtype}")

    size = struct.unpack("<Q", blob[5:13])[0]

    expected_data_len = size * 4  # float32 is 4 bytes
    actual_data = blob[13:]
    if len(actual_data) != expected_data_len:
        raise ValueError(
            f"Data size mismatch: expected {expected_data_len}, got {len(actual_data)}"
        )

    floats = np.frombuffer(actual_data, dtype=np.float32)
    return floats.tolist()
