"""Inspect a TensorFlow checkpoint and print all variable names and shapes."""

import sys
import os

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

def main():
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Src/Ext/handwriting-synthesis/model/checkpoint"
    )

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint is None:
        print(f"No checkpoint found in {checkpoint_dir}")
        sys.exit(1)

    print(f"Checkpoint: {checkpoint}")
    print()

    reader = tf.train.load_checkpoint(checkpoint)
    var_to_shape = reader.get_variable_to_shape_map()
    var_to_dtype = reader.get_variable_to_dtype_map()

    for name in sorted(var_to_shape.keys()):
        shape = var_to_shape[name]
        dtype = var_to_dtype[name]
        print(f"{name:60s} {str(shape):20s} {dtype.name}")


if __name__ == "__main__":
    main()
