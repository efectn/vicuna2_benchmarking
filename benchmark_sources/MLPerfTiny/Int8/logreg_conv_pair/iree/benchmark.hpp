#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include "logreg_conv_pair_int8_input_data.h"
#include "logreg_conv_pair_int8_model_settings.h"
#include "logreg_conv_pair_int8_output_data_ref.h"
#include "simulator.hpp"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

#include <stdlib.h>
#include <stdint.h>
extern "C" {
extern const unsigned char iree_model_vmfb[];
extern const unsigned int iree_model_vmfb_len;
}

class Benchmark {
  private:
    iree_runtime_instance_t* instance = nullptr;
    iree_runtime_session_t* session = nullptr;
    iree_runtime_call_t call;

  public:
    Benchmark() {
        iree_allocator_t allocator = iree_allocator_system();

        iree_runtime_instance_options_t instance_options;
        iree_runtime_instance_options_initialize(&instance_options);
        iree_runtime_instance_options_use_all_available_drivers(&instance_options);

        iree_status_t status = iree_runtime_instance_create(&instance_options, allocator, &instance);
        if (!iree_status_is_ok(status)) return;

        iree_hal_device_t* device = nullptr;
        status = iree_runtime_instance_try_create_default_device(
            instance, iree_make_cstring_view("local-sync"), &device);
        if (!iree_status_is_ok(status)) return;

        iree_runtime_session_options_t session_options;
        iree_runtime_session_options_initialize(&session_options);
        status = iree_runtime_session_create_with_device(
            instance, &session_options, device, allocator, &session);
        iree_hal_device_release(device);
        if (!iree_status_is_ok(status)) return;

        iree_vm_module_t* module = nullptr;
        status = iree_vm_bytecode_module_create(
            iree_runtime_instance_vm_instance(instance), IREE_VM_BYTECODE_MODULE_FLAG_NONE,
            iree_make_const_byte_span(iree_model_vmfb, iree_model_vmfb_len),
            iree_allocator_null(), allocator, &module);
        if (!iree_status_is_ok(status)) return;

        status = iree_runtime_session_append_module(session, module);
        iree_vm_module_release(module);
        if (!iree_status_is_ok(status)) return;

        status = iree_runtime_call_initialize_by_name(
            session, iree_make_cstring_view("module.main"), &call);
    }

    inline int run_benchmark() {
        Simulator simulator;
        if (!instance || !session) {
            simulator.terminate(10);
            return 10;
        }

        iree_hal_allocator_t* device_allocator = iree_runtime_session_device_allocator(session);
        iree_hal_dim_t shape[4];
        for (size_t i = 0; i < logreg_conv_pair_int8_input_rank; ++i) {
            shape[i] = (iree_hal_dim_t)logreg_conv_pair_int8_input_shape[i];
        }

        iree_hal_buffer_view_t* arg0 = nullptr;
        iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
            iree_runtime_session_device(session), device_allocator,
            logreg_conv_pair_int8_input_rank, shape, IREE_HAL_ELEMENT_TYPE_SINT_8,
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
            (iree_hal_buffer_params_t){.usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
                                       .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL},
            iree_make_const_byte_span(logreg_conv_pair_int8_input_data[0],
                                      logreg_conv_pair_int8_input_data_len[0]),
            &arg0);

        if (!iree_status_is_ok(status)) {
            simulator.terminate(11);
            return 11;
        }
        iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
        iree_hal_buffer_view_release(arg0);

        simulator.begin_measurement();
        status = iree_runtime_call_invoke(&call, 0);
        simulator.end_measurement();

        if (!iree_status_is_ok(status)) {
            simulator.terminate(12);
            return 12;
        }

        iree_hal_buffer_view_t* ret0 = nullptr;
        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret0);
        if (!iree_status_is_ok(status) || !ret0) {
            simulator.terminate(13);
            return 13;
        }

        iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(ret0);
        iree_hal_buffer_mapping_t mapping;
        int result = 0;

        if (iree_status_is_ok(iree_hal_buffer_map_range(
                buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
                0, logreg_conv_pair_int8_output_data_ref_len, &mapping))) {
            const int8_t* out_ptr = (const int8_t*)mapping.contents.data;
            for (size_t j = 0; j < logreg_conv_pair_int8_output_data_ref_len; ++j) {
                if (abs((int)out_ptr[j] - (int)logreg_conv_pair_int8_output_data_ref[j]) > 1) {
                    simulator.terminate((int)j + 1);
                    result = (int)j + 1;
                    break;
                }
            }
            if (result == 0) simulator.terminate(0);
            iree_hal_buffer_unmap_range(&mapping);
        } else {
            simulator.terminate(14);
            result = 14;
        }

        iree_hal_buffer_view_release(ret0);
        return result;
    }

    int validate_benchmark() { return 0; }

    ~Benchmark() {
        iree_runtime_call_deinitialize(&call);
        if (session) iree_runtime_session_release(session);
        if (instance) iree_runtime_instance_release(instance);
    }
};

#endif /* BENCHMARK_HPP */
