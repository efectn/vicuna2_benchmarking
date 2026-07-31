module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\0C\00\00\00\08\00\0E\00\08\00\04\00\08\00\00\00\10\00\00\00(\00\00\00\00\00\06\00\08\00\04\00\06\00\00\00\04\00\00\00\01\00\00\00\EB\03\00\00\0C\00\1C\00\18\00\14\00\10\00\04\00\0C\00\00\00\03O:5z\B2\E1\04\00\00\00\00\02\00\00\00\02\00\00\00\04\00\00\00\12\00\00\002.22.0-dev20260726\00\00", min_runtime_version = "1.14.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x4xi8> {tf_saved_model.index_path = ["input"]}) -> (tensor<1x8xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_input:0", outputs = "StatefulPartitionedCall_1:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.const_shape  {values = dense<[1, 8]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = "tosa.const"() <{values = dense<[40, 41, 38, 39, 38, 39, 41, 38]> : tensor<8xi8>}> : () -> tensor<8xi8>
    %2 = "tosa.const"() <{values = dense<[2046561905, 1624229687, 1155463991, 1677328400, 1325728765, 1641204169, 2123091233, 1210512434]> : tensor<8xi32>}> : () -> tensor<8xi32>
    %3 = "tosa.const"() <{values = dense<-29> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = tosa.const_shape  {values = dense<[8, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
    %6 = "tosa.const"() <{values = dense<[[127], [127], [127], [-127], [127], [-127], [127], [127]]> : tensor<8x1xi8>}> : () -> tensor<8x1xi8>
    %7 = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
    %8 = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
    %9 = "tosa.const"() <{values = dense<1183257103> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %11 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %12 = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %13 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = "tosa.const"() <{values = dense<[[-74, -93, 86, -127]]> : tensor<1x4xi8>}> : () -> tensor<1x4xi8>
    %15 = tosa.const_shape  {values = dense<[1, 1, 1, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %16 = tosa.reshape %arg0, %15 : (tensor<1x4xi8>, !tosa.shape<4>) -> tensor<1x1x1x4xi8>
    %17 = tosa.reshape %14, %15 : (tensor<1x4xi8>, !tosa.shape<4>) -> tensor<1x1x1x4xi8>
    %18 = tosa.conv2d %16, %17, %13, %12, %11 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x4xi8>, tensor<1x1x1x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x1x1xi32>
    %19 = tosa.rescale %18, %9, %8, %13, %10 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x1x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x1x1xi8>
    %20 = tosa.reshape %19, %7 : (tensor<1x1x1x1xi8>, !tosa.shape<2>) -> tensor<1x1xi8>
    %21 = tosa.clamp %20 {max_val = 127 : i8, min_val = -128 : i8} : (tensor<1x1xi8>) -> tensor<1x1xi8>
    %22 = tosa.reshape %21, %5 : (tensor<1x1xi8>, !tosa.shape<4>) -> tensor<1x1x1x1xi8>
    %23 = tosa.reshape %6, %4 : (tensor<8x1xi8>, !tosa.shape<4>) -> tensor<8x1x1x1xi8>
    %24 = tosa.conv2d %22, %23, %13, %10, %11 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x1x1x1xi8>, tensor<8x1x1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x1x8xi32>
    %25 = tosa.rescale %24, %2, %1, %13, %3 {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x1x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x1x8xi8>
    %26 = tosa.reshape %25, %0 : (tensor<1x1x1x8xi8>, !tosa.shape<2>) -> tensor<1x8xi8>
    return %26 : tensor<1x8xi8>
  }
}
