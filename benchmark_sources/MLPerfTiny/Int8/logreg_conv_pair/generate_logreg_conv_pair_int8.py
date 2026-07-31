#!/usr/bin/env python3
"""Generate a single fully-connected int8 model and TOSA MLIR."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
IREE_DIR = ROOT / "iree"

MODEL_NAME = "logreg_conv_pair_int8"
INPUT_SIZE = 4
HIDDEN_SIZE = 1
OUTPUT_SIZE = 8
# working patterns: 4 2/4 1/2
# not working patterns: 4 2 8, 4 1 8
VENV_BIN = Path(sys.prefix) / "bin"
TOSA_CONVERTER_FOR_TFLITE = VENV_BIN / "tosa-converter-for-tflite"


def c_array(values: np.ndarray, per_line: int = 12) -> str:
    items = values.reshape(-1).tolist()

    lines = []
    for index in range(0, len(items), per_line):
        line_items = items[index : index + per_line]
        lines.append("  " + ", ".join(str(int(x)) for x in line_items) + ",")

    return "\n".join(lines)


def build_tflite() -> bytes:
    np.random.seed(29)
    tf.random.set_seed(29)

    model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(INPUT_SIZE,),
            batch_size=1,
            name="input",
        ),
        tf.keras.layers.Dense(
            HIDDEN_SIZE,
            activation="relu",
            name="fc2",
        ),
        tf.keras.layers.Dense(
            OUTPUT_SIZE,
            activation=None,
            name="output",
        ),
    ]
)

    model(np.zeros((1, INPUT_SIZE), dtype=np.float32))

    def representative_dataset():
        indices = np.arange(INPUT_SIZE, dtype=np.float32)

        samples = [
            np.linspace(-1.0, 1.0, INPUT_SIZE, dtype=np.float32),
            np.linspace(1.0, -1.0, INPUT_SIZE, dtype=np.float32),
            np.sin(indices * 0.31).astype(np.float32),
            np.cos(indices * 0.37).astype(np.float32),
            np.zeros(INPUT_SIZE, dtype=np.float32),
            np.full(INPUT_SIZE, 0.5, dtype=np.float32),
        ]

        for sample in samples:
            yield [sample.reshape(1, INPUT_SIZE)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT,
    ]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def run_reference(
    tflite_model: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]

    sample = (
        np.sin(np.arange(INPUT_SIZE, dtype=np.float32) * 0.31) * 0.75
    ).reshape(1, INPUT_SIZE)

    quantized_input = np.round(
        sample / input_scale + input_zero_point
    ).clip(-128, 127).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        quantized_input,
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details["index"]
    ).astype(np.int8)

    return (
        quantized_input.reshape(-1),
        output.reshape(-1),
    )


def write_inputs(
    tflite_model: bytes,
    input_data: np.ndarray,
    output_ref: np.ndarray,
) -> None:
    data_dir = ROOT / f"{MODEL_NAME}_data"

    IREE_DIR.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    tflite_path = ROOT / f"{MODEL_NAME}.tflite"
    input_txt_path = ROOT / f"{MODEL_NAME}_input.txt"
    output_txt_path = ROOT / f"{MODEL_NAME}_output_ref.txt"

    tflite_path.write_bytes(tflite_model)

    input_txt_path.write_text(
        " ".join(str(int(x)) for x in input_data) + "\n",
        encoding="utf-8",
    )

    output_txt_path.write_text(
        " ".join(str(int(x)) for x in output_ref) + "\n",
        encoding="utf-8",
    )

    input_header = data_dir / f"{MODEL_NAME}_input_data.h"
    input_source = data_dir / f"{MODEL_NAME}_input_data.cc"

    input_header.write_text(
        f"""#ifndef SINGLE_DENSE_INT8_INPUT_DATA_H
#define SINGLE_DENSE_INT8_INPUT_DATA_H

#include <stddef.h>
#include <stdint.h>

extern const size_t {MODEL_NAME}_data_sample_cnt;
extern const int8_t* {MODEL_NAME}_input_data[];
extern const size_t {MODEL_NAME}_input_data_len[];

#endif
""",
        encoding="utf-8",
    )

    input_source.write_text(
        f"""#include "{MODEL_NAME}_input_data.h"

const int8_t {MODEL_NAME}_input_data_0000[] = {{
{c_array(input_data)}
}};

const size_t {MODEL_NAME}_input_data_0000_len =
    {input_data.size};

const size_t {MODEL_NAME}_data_sample_cnt = 1;

const int8_t* {MODEL_NAME}_input_data[] = {{
    {MODEL_NAME}_input_data_0000
}};

const size_t {MODEL_NAME}_input_data_len[] = {{
    {MODEL_NAME}_input_data_0000_len
}};
""",
        encoding="utf-8",
    )

    output_header = data_dir / f"{MODEL_NAME}_output_data_ref.h"
    output_source = data_dir / f"{MODEL_NAME}_output_data_ref.cc"

    output_header.write_text(
        f"""#ifndef SINGLE_DENSE_INT8_OUTPUT_DATA_REF_H
#define SINGLE_DENSE_INT8_OUTPUT_DATA_REF_H

#include <stddef.h>
#include <stdint.h>

extern const int8_t {MODEL_NAME}_output_data_ref[];
extern const size_t {MODEL_NAME}_output_data_ref_len;

#endif
""",
        encoding="utf-8",
    )

    output_source.write_text(
        f"""#include "{MODEL_NAME}_output_data_ref.h"

const int8_t {MODEL_NAME}_output_data_ref[] = {{
{c_array(output_ref)}
}};

const size_t {MODEL_NAME}_output_data_ref_len =
    {output_ref.size};
""",
        encoding="utf-8",
    )

    settings_header = data_dir / f"{MODEL_NAME}_model_settings.h"
    settings_source = data_dir / f"{MODEL_NAME}_model_settings.cc"

    settings_header.write_text(
        f"""#ifndef SINGLE_DENSE_INT8_MODEL_SETTINGS_H
#define SINGLE_DENSE_INT8_MODEL_SETTINGS_H

#include <stddef.h>
#include <stdint.h>

extern const size_t {MODEL_NAME}_input_feature_cnt;
extern const size_t {MODEL_NAME}_output_feature_cnt;
extern const size_t {MODEL_NAME}_input_rank;
extern const size_t {MODEL_NAME}_input_shape[2];

#endif
""",
        encoding="utf-8",
    )

    settings_source.write_text(
        f"""#include "{MODEL_NAME}_model_settings.h"

const size_t {MODEL_NAME}_input_feature_cnt = {INPUT_SIZE};
const size_t {MODEL_NAME}_output_feature_cnt = {OUTPUT_SIZE};
const size_t {MODEL_NAME}_input_rank = 2;

const size_t {MODEL_NAME}_input_shape[2] = {{
    1,
    {INPUT_SIZE}
}};
""",
        encoding="utf-8",
    )


def write_mlir(tflite_path: Path) -> None:
    mlir_path = IREE_DIR / f"{MODEL_NAME}_model.mlir"

    subprocess.run(
        [
            str(TOSA_CONVERTER_FOR_TFLITE),
            str(tflite_path),
            "--text",
            "-o",
            str(mlir_path),
        ],
        check=True,
    )


def main() -> None:
    tflite_model = build_tflite()

    input_data, output_ref = run_reference(tflite_model)

    write_inputs(
        tflite_model,
        input_data,
        output_ref,
    )

    tflite_path = ROOT / f"{MODEL_NAME}.tflite"
    mlir_path = IREE_DIR / f"{MODEL_NAME}_model.mlir"

    write_mlir(tflite_path)

    print(f"Model architecture: {INPUT_SIZE} -> {OUTPUT_SIZE}")
    print(f"Wrote {tflite_path}")
    print(f"Wrote {mlir_path}")
    print(f"Reference input:  {input_data.tolist()}")
    print(f"Reference output: {output_ref.tolist()}")


if __name__ == "__main__":
    main()