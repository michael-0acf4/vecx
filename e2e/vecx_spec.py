import struct
import numpy as np


class VECX_DTYPE:
    INT32 = 1
    INT64 = 2
    FLOAT32 = 3


def pack_vecx_f32_blob(floats):
    size = len(floats)
    blob = bytearray()
    blob += b"vecx"  # 4 bytes
    blob += struct.pack("<B", VECX_DTYPE.FLOAT32)  # 1 byte
    blob += struct.pack("<Q", size)  # 8 bytes
    blob += np.array(floats, dtype=np.float32).tobytes()
    return blob
