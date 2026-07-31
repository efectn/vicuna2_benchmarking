#ifndef LOGREG_INT8_MODEL_SETTINGS_H
#define LOGREG_INT8_MODEL_SETTINGS_H

#include <stddef.h>
#include <stdint.h>

const size_t logreg_int8_input_feature_cnt = 64;
const size_t logreg_int8_output_feature_cnt = 112;
const size_t logreg_int8_input_rank = 4;
const size_t logreg_int8_input_shape[4] = {1, 8, 8, 1};

#endif /* LOGREG_INT8_MODEL_SETTINGS_H */
