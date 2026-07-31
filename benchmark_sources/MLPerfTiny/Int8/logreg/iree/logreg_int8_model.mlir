module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\0C\00\00\00\08\00\0E\00\08\00\04\00\08\00\00\00\10\00\00\00(\00\00\00\00\00\06\00\08\00\04\00\06\00\00\00\04\00\00\00\01\00\00\00\EB\03\00\00\0C\00\1C\00\18\00\14\00\10\00\04\00\0C\00\00\00\86\8F\F6n\F4i\17\04\00\00\00\00\02\00\00\00\02\00\00\00\04\00\00\00\12\00\00\002.22.0-dev20260726\00\00", min_runtime_version = "1.14.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<?x8x8x1xi8> {tf_saved_model.index_path = ["image"]}) -> (tensor<?x8x7x2xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_image:0", outputs = "StatefulPartitionedCall_1:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<[37, 38]> : tensor<2xi8>}> : () -> tensor<2xi8>
    %1 = "tosa.const"() <{values = dense<2058215333> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2 = "tosa.const"() <{values = dense<-25> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "tosa.const"() <{values = dense<[[[[-127], [64]]], [[[64], [-127]]]]> : tensor<2x1x2x1xi8>}> : () -> tensor<2x1x2x1xi8>
    %5 = "tosa.const"() <{values = dense<[1799, -5398]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %6 = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = tosa.conv2d %arg0, %4, %5, %6, %7 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x8x8x1xi8>, tensor<2x1x2x1xi8>, tensor<2xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x8x7x2xi32>
    %9 = tosa.rescale %8, %1, %0, %3, %2 {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<?x8x7x2xi32>, tensor<2xi32>, tensor<2xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<?x8x7x2xi8>
    return %9 : tensor<?x8x7x2xi8>
  }
}
