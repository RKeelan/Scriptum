"""Run the TF model on a fixed input and dump intermediate values to JSON.

This provides reference values for validating the TypeScript forward pass.
We run a few timesteps deterministically (no sampling — just the forward pass
with fixed input) and dump the intermediate states.
"""

import json
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfcompat

# Add the handwriting-synthesis repo to the path
HS_PATH = os.path.expanduser("~/Src/Ext/handwriting-synthesis")
sys.path.insert(0, HS_PATH)

tfcompat.disable_v2_behavior()

from handwriting_synthesis.drawing.operations import alphabet, alpha_to_num
from handwriting_synthesis.rnn.LSTMAttentionCell import LSTMAttentionCell
from handwriting_synthesis.tf.utils import dense_layer


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "reference-values.json",
    )

    text = "hi"
    num_steps = 5
    bias_value = 0.5

    # Encode text
    char_indices = list(map(lambda x: alpha_to_num[x], text)) + [0]
    char_len = len(char_indices)
    alphabet_size = len(alphabet)

    print(f"Text: '{text}'")
    print(f"Char indices: {char_indices}")
    print(f"Char len: {char_len}")
    print(f"Alphabet size: {alphabet_size}")

    # Build the graph
    tfcompat.reset_default_graph()

    c_placeholder = tfcompat.placeholder(tf.int32, [1, None])
    c_len_placeholder = tfcompat.placeholder(tf.int32, [1])
    bias_placeholder = tfcompat.placeholder(tf.float32, [1])

    cell = LSTMAttentionCell(
        lstm_size=400,
        num_attn_mixture_components=10,
        attention_values=tf.one_hot(c_placeholder, alphabet_size),
        attention_values_lengths=c_len_placeholder,
        num_output_mixture_components=20,
        bias=bias_placeholder,
    )

    input_ph = tfcompat.placeholder(tf.float32, [1, 3])

    with tfcompat.variable_scope("rnn"):
        state = cell.zero_state(1, dtype=tf.float32)
        output_op, new_state_op = cell(input_ph, state)
        gmm_params = dense_layer(new_state_op.h3, cell.output_units, scope="gmm", reuse=tfcompat.AUTO_REUSE)

    # Load checkpoint
    checkpoint_dir = os.path.join(HS_PATH, "model", "checkpoint")
    saver = tfcompat.train.Saver()

    results = []

    with tfcompat.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, checkpoint)

        # Prepare character input
        chars = np.zeros([1, 120], dtype=np.int32)
        chars[0, :char_len] = char_indices

        # Initial input: pen up
        x = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        current_state = sess.run(state, feed_dict={
            c_placeholder: chars[:, :char_len],
            c_len_placeholder: [char_len],
            bias_placeholder: [bias_value],
        })

        for step in range(num_steps):
            feed = {
                input_ph: x,
                c_placeholder: chars[:, :char_len],
                c_len_placeholder: [char_len],
                bias_placeholder: [bias_value],
                state: current_state,
            }

            out, ns, gmm = sess.run([output_op, new_state_op, gmm_params], feed_dict=feed)

            step_data = {
                "step": step,
                "input": x[0].tolist(),
                "h1": ns.h1[0].tolist(),
                "c1": ns.c1[0].tolist(),
                "h2": ns.h2[0].tolist(),
                "h3": ns.h3[0].tolist(),
                "kappa": ns.kappa[0].tolist(),
                "window": ns.w[0].tolist(),
                "gmm_raw": gmm[0].tolist(),
            }
            results.append(step_data)

            current_state = ns

            # Use a fixed next input (no sampling — deterministic)
            x = np.array([[0.1 * (step + 1), -0.05 * (step + 1), 0.0]], dtype=np.float32)

            print(f"Step {step}: h1 norm={np.linalg.norm(ns.h1):.6f}, "
                  f"window norm={np.linalg.norm(ns.w):.6f}, "
                  f"kappa mean={np.mean(ns.kappa):.6f}")

    output = {
        "text": text,
        "char_indices": char_indices,
        "bias": bias_value,
        "num_steps": num_steps,
        "steps": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
