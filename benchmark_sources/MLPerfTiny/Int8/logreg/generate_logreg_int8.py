#!/usr/bin/env python3
"""Generate a minimal int8 convolutional TFLite model and TOSA MLIR."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
IREE_DIR = ROOT / "iree"
MODEL_NAME = "logreg_int8"
VENV_BIN = Path(sys.prefix) / "bin"
TOSA_CONVERTER_FOR_TFLITE = VENV_BIN / "tosa-converter-for-tflite"


def c_array(values: bytes | np.ndarray, per_line: int = 12, hex_bytes: bool = False) -> str:
    if isinstance(values, np.ndarray):
        items = values.reshape(-1).tolist()
    else:
        items = list(values)

    rendered: list[str] = []
    for i in range(0, len(items), per_line):
        line_items = items[i : i + per_line]
        if hex_bytes:
            rendered.append("  " + ", ".join(f"0x{x:02x}" for x in line_items) + ",")
        else:
            rendered.append("  " + ", ".join(str(int(x)) for x in line_items) + ",")
    return "\n".join(rendered)


def build_tflite() -> bytes:
    np.random.seed(7)
    tf.random.set_seed(7)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8, 8, 1), name="image"),
            tf.keras.layers.Conv2D(
                2,
                kernel_size=(1, 2),
                padding="valid",
                activation=None,
                use_bias=True,
                name="conv",
            ),
        ]
    )
    model(np.zeros((1, 8, 8, 1), dtype=np.float32))

    conv_weights = np.array(
        [
            [[[-0.20, 0.05]], [[0.10, -0.10]]],
        ],
        dtype=np.float32,
    )
    conv_bias = np.array([0.02, -0.03], dtype=np.float32)
    model.get_layer("conv").set_weights([conv_weights, conv_bias])

    def representative_dataset():
        base = np.linspace(-0.9, 0.9, 64, dtype=np.float32).reshape(8, 8, 1)
        samples = np.stack(
            [
                base,
                np.flip(base, axis=0),
                np.sin(np.arange(64, dtype=np.float32) * 0.25).reshape(8, 8, 1) * 0.8,
                np.zeros((8, 8, 1), dtype=np.float32),
            ],
            axis=0,
        )
        for sample in samples:
            yield [sample.reshape(1, 8, 8, 1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def run_reference(tflite_model: bytes) -> tuple[np.ndarray, np.ndarray]:
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    scale, zero_point = input_details["quantization"]

    sample = np.sin(np.arange(64, dtype=np.float32) * 0.25).reshape(1, 8, 8, 1) * 0.8
    quantized = np.round(sample / scale + zero_point).clip(-128, 127).astype(np.int8)

    interpreter.set_tensor(input_details["index"], quantized)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"]).astype(np.int8)
    return quantized.reshape(-1), output.reshape(-1)


def write_inputs(tflite_model: bytes, input_data: np.ndarray, output_ref: np.ndarray) -> None:
    data_dir = ROOT / f"{MODEL_NAME}_data"
    IREE_DIR.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / f"{MODEL_NAME}.tflite").write_bytes(tflite_model)
    (ROOT / f"{MODEL_NAME}_input.txt").write_text(
        " ".join(str(int(x)) for x in input_data.tolist()) + "\n",
        encoding="utf-8",
    )
    (ROOT / f"{MODEL_NAME}_output_ref.txt").write_text(
        " ".join(str(int(x)) for x in output_ref.tolist()) + "\n",
        encoding="utf-8",
    )
    (data_dir / f"{MODEL_NAME}_input_data.h").write_text(
        f"""#ifndef LOGREG_INT8_INPUT_DATA_H
#define LOGREG_INT8_INPUT_DATA_H

#include <stddef.h>
#include <stdint.h>

const size_t {MODEL_NAME}_data_sample_cnt = 1;
extern const int8_t* {MODEL_NAME}_input_data[];
extern const size_t {MODEL_NAME}_input_data_len[];

#endif /* LOGREG_INT8_INPUT_DATA_H */
""",
        encoding="utf-8",
    )
    (data_dir / f"{MODEL_NAME}_input_data.cc").write_text(
        f"""#include "{MODEL_NAME}_input_data.h"

const int8_t {MODEL_NAME}_input_data_0000[] = {{
{c_array(input_data)}
}};

const size_t {MODEL_NAME}_input_data_0000_len = {input_data.size};
const int8_t* {MODEL_NAME}_input_data[] = {{{MODEL_NAME}_input_data_0000}};
const size_t {MODEL_NAME}_input_data_len[] = {{{MODEL_NAME}_input_data_0000_len}};
""",
        encoding="utf-8",
    )
    (data_dir / f"{MODEL_NAME}_output_data_ref.h").write_text(
        f"""#ifndef LOGREG_INT8_OUTPUT_DATA_REF_H
#define LOGREG_INT8_OUTPUT_DATA_REF_H

#include <stddef.h>
#include <stdint.h>

extern const int8_t {MODEL_NAME}_output_data_ref[];
extern const size_t {MODEL_NAME}_output_data_ref_len;

#endif /* LOGREG_INT8_OUTPUT_DATA_REF_H */
""",
        encoding="utf-8",
    )
    (data_dir / f"{MODEL_NAME}_output_data_ref.cc").write_text(
        f"""#include "{MODEL_NAME}_output_data_ref.h"

const int8_t {MODEL_NAME}_output_data_ref[] = {{
{c_array(output_ref)}
}};

const size_t {MODEL_NAME}_output_data_ref_len = {output_ref.size};
""",
        encoding="utf-8",
    )
    (data_dir / f"{MODEL_NAME}_model_settings.h").write_text(
            f"""#ifndef LOGREG_INT8_MODEL_SETTINGS_H
#define LOGREG_INT8_MODEL_SETTINGS_H

#include <stddef.h>
#include <stdint.h>

const size_t logreg_int8_input_feature_cnt = {input_data.size};
const size_t logreg_int8_output_feature_cnt = {output_ref.size};
const size_t logreg_int8_input_rank = 4;
const size_t logreg_int8_input_shape[4] = {{1, 8, 8, 1}};

#endif /* LOGREG_INT8_MODEL_SETTINGS_H */
""",
        encoding="utf-8",
    )
    (data_dir / f"{MODEL_NAME}_model_settings.cc").write_text(
        f"""#include "{MODEL_NAME}_model_settings.h"
""",
        encoding="utf-8",
    )


def write_mlir(tflite_path: Path) -> None:
    mlir = IREE_DIR / f"{MODEL_NAME}_model.mlir"
    subprocess.run(
        [str(TOSA_CONVERTER_FOR_TFLITE), str(tflite_path), "--text", "-o", str(mlir)],
        check=True,
    )


def main() -> None:
    tflite_model = build_tflite()
    input_data, output_ref = run_reference(tflite_model)
    write_inputs(tflite_model, input_data, output_ref)
    write_mlir(ROOT / f"{MODEL_NAME}.tflite")
    print(f"Wrote {ROOT / f'{MODEL_NAME}.tflite'}")
    print(f"Wrote {IREE_DIR / f'{MODEL_NAME}_model.mlir'}")
    print(f"Input int8: {input_data.tolist()}")
    print(f"Reference int8 output: {output_ref.tolist()}")


if __name__ == "__main__":
    main()
