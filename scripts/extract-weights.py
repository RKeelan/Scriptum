"""Extract weights from TensorFlow checkpoint into a binary format.

Binary format:
  Header: magic "SCRP" (4 bytes), tensor_count (uint32 LE)
  Per tensor:
    name_length (uint16 LE)
    name (utf8 bytes)
    ndims (uint8)
    shape (ndims x uint32 LE)
    data (float32 LE[product(shape)])
"""

import os
import struct
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

# Map from TF variable names to our logical names
WEIGHT_MAP = {
    "rnn/LSTMAttentionCell/lstm_cell/kernel": "lstm1_kernel",
    "rnn/LSTMAttentionCell/lstm_cell/bias": "lstm1_bias",
    "rnn/LSTMAttentionCell/attention/weights": "attention_weights",
    "rnn/LSTMAttentionCell/attention/biases": "attention_bias",
    "rnn/LSTMAttentionCell/lstm_cell_1/kernel": "lstm2_kernel",
    "rnn/LSTMAttentionCell/lstm_cell_1/bias": "lstm2_bias",
    "rnn/LSTMAttentionCell/lstm_cell_2/kernel": "lstm3_kernel",
    "rnn/LSTMAttentionCell/lstm_cell_2/bias": "lstm3_bias",
    "rnn/gmm/weights": "gmm_weights",
    "rnn/gmm/biases": "gmm_bias",
}

def main():
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Src/Ext/handwriting-synthesis/model/checkpoint"
    )
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "weights",
        "synthesis.bin",
    )

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint is None:
        print(f"No checkpoint found in {checkpoint_dir}")
        sys.exit(1)

    print(f"Checkpoint: {checkpoint}")
    reader = tf.train.load_checkpoint(checkpoint)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Kernel tensors need transposing: TF stores [input, output] but we want
    # [output, input] row-major for efficient matVecMul(kernel, input, output, input).
    TRANSPOSE = {
        "lstm1_kernel", "lstm2_kernel", "lstm3_kernel",
        "attention_weights", "gmm_weights",
    }

    tensors = []
    for tf_name, logical_name in sorted(WEIGHT_MAP.items()):
        array = reader.get_tensor(tf_name).astype(np.float32)
        if logical_name in TRANSPOSE and array.ndim == 2:
            array = array.T.copy()
        tensors.append((logical_name, array.shape, array))
        total = int(np.prod(array.shape))
        print(
            f"  {logical_name:25s} {str(array.shape):20s} "
            f"{total * 4:>10,} bytes"
        )

    with open(output_path, "wb") as f:
        # Header
        f.write(b"SCRP")
        f.write(struct.pack("<I", len(tensors)))

        for name, shape, data in tensors:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", len(shape)))
            for dim in shape:
                f.write(struct.pack("<I", dim))
            f.write(data.tobytes())

    file_size = os.path.getsize(output_path)
    print(f"\nWrote {output_path} ({file_size:,} bytes, {file_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
