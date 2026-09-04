#ifndef AADESH_OPS
#define AADESH_OPS

include "Aadesh/AadeshDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributes.td"

class Aadesh_Op<string mnemonic, list<Trait> traits = []> :
    Op<Aadesh_Dialect, mnemonic, traits>;

// FloatElementsAttr<width> class exists upstream (CommonAttrConstraints.td)
// but only F64ElementsAttr is instantiated from it — F32 is missing, so we
// define it ourselves here.
def F32ElementsAttr : FloatElementsAttr<32>;

//===----------------------------------------------------------------------===//
// AddOp — existing op, unchanged
//===----------------------------------------------------------------------===//

def AddOp : Aadesh_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Addition operation in aadesh dialect";

  let arguments = (ins
    AnyTypeOf<[F32, F64, AnySignlessInteger, AnySignedInteger, AnyUnsignedInteger,
               TensorOf<[F32, F64, AnyInteger]>]>:$lhs,
    AnyTypeOf<[F32, F64, AnySignlessInteger, AnySignedInteger, AnyUnsignedInteger,
               TensorOf<[F32, F64, AnyInteger]>]>:$rhs
  );

  let results = (outs
    AnyTypeOf<[F32, F64, AnySignlessInteger, AnySignedInteger, AnyUnsignedInteger,
               TensorOf<[F32, F64, AnyInteger]>]>:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// ApplyMatrixOp — new op, added alongside AddOp
//===----------------------------------------------------------------------===//

def ApplyMatrixOp : Aadesh_Op<"apply_matrix", [Pure]> {
  let summary = "Applies a 2D float matrix attribute to one or two input tensors";
  let description = [{
    `input1` is required. `input2` is optional. Both accept tensors of
    float or integer element type. `matrix` is a compile-time f32 dense
    elements constant attached to the op; F32ElementsAttr checks element
    type but not rank/shape, so rank == 2 is checked in the verifier.
    `result` is always a rank-3 float tensor.
  }];

  let arguments = (ins
    TensorOf<[AnyFloat, AnyInteger]>:$input1,
    Optional<TensorOf<[AnyFloat, AnyInteger]>>:$input2,
    F32ElementsAttr:$matrix
  );

  let results = (outs
    TensorRankOf<[AnyFloat], [3]>:$result
  );

  let assemblyFormat = [{
    $input1 (`,` $input2^)? attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

#endif // AADESH_OPS


//===----------------------------------------------------------------------===//
// MulOp — new op, mirrors AddOp's type coverage
//===----------------------------------------------------------------------===//

def MulOp : Aadesh_Op<"mul", [Pure, SameOperandsAndResultType]> {
  let summary = "Multiplication operation in aadesh dialect";

  let arguments = (ins
    AnyTypeOf<[F32, F64, AnySignlessInteger, AnySignedInteger, AnyUnsignedInteger,
               TensorOf<[F32, F64, AnyInteger]>]>:$lhs,
    AnyTypeOf<[F32, F64, AnySignlessInteger, AnySignedInteger, AnyUnsignedInteger,
               TensorOf<[F32, F64, AnyInteger]>]>:$rhs
  );

  let results = (outs
    AnyTypeOf<[F32, F64, AnySignlessInteger, AnySignedInteger, AnyUnsignedInteger,
               TensorOf<[F32, F64, AnyInteger]>]>:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// ReluOp — new op, mirrors AddOp's type coverage (single operand)
//===----------------------------------------------------------------------===//

def ReluOp : Aadesh_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Rectified linear unit: returns max(0, x)";
  let description = [{
    Elementwise `max(0, x)`. Supports the same scalar float/integer types
    and float/integer tensors as `aadesh.add`/`aadesh.mul`. Note that for
    unsigned integer operands `max(0, x) == x` is a no-op by definition,
    but the type is still accepted for consistency with the rest of the
    dialect's type coverage.
  }];

  let arguments = (ins
    AnyTypeOf<[F32, F64, AnySignlessInteger, AnySignedInteger, AnyUnsignedInteger,
               TensorOf<[F32, F64, AnyInteger]>]>:$input
  );

  let results = (outs
    AnyTypeOf<[F32, F64, AnySignlessInteger, AnySignedInteger, AnyUnsignedInteger,
               TensorOf<[F32, F64, AnyInteger]>]>:$result
  );

  let assemblyFormat = "$input attr-dict `:` type($result)";
}
