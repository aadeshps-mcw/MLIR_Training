#include "Aadesh/AadeshOps.h"
#include "Aadesh/AadeshDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#define GET_OP_CLASSES
#include "Aadesh/AadeshOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ApplyMatrixOp::verify()
//===----------------------------------------------------------------------===//
//
// ODS (hasVerifier = 1) only declares this function in the generated
// header; it does not implement it. `matrix` is a DenseFPElementsAttr
// (F32ElementsAttr), which is backed by a RankedTensorType — so unlike
// the old nested ArrayAttr representation, ragged rows are impossible
// by construction. The only thing ODS can't express is rank, so that's
// checked here.

::llvm::LogicalResult mlir::aadesh::ApplyMatrixOp::verify()
{
  mlir::DenseElementsAttr matrix = getMatrix();
  mlir::ShapedType matrixType = matrix.getType();

  if (matrixType.getRank() != 2)
    return emitOpError("matrix attribute must be a rank-2 (2D) tensor, got rank ")
           << matrixType.getRank();

  return ::mlir::success();
}

// Computes the NumPy-broadcast output shape of two shapes, or failure if
// incompatible. Only handles fully static shapes for now -- dynamic-shape
// broadcasting is not yet supported by this verifier/lowering pair.
static mlir::FailureOr<llvm::SmallVector<int64_t>>
computeBroadcastShape(llvm::ArrayRef<int64_t> lhs,
                      llvm::ArrayRef<int64_t> rhs)
{
  size_t rank = std::max(lhs.size(), rhs.size());
  llvm::SmallVector<int64_t> result(rank);

  for (size_t i = 0; i < rank; ++i)
  {
    int64_t l =
        i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
    int64_t r =
        i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];

    if (l != r && l != 1 && r != 1)
      return mlir::failure();

    result[i] = std::max(l, r);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// PowOp::verify()
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::aadesh::PowOp::verify()
{
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  auto lhsTensor = llvm::dyn_cast<RankedTensorType>(lhsType);
  auto rhsTensor = llvm::dyn_cast<RankedTensorType>(rhsType);
  auto resultTensor = llvm::dyn_cast<RankedTensorType>(resultType);

  if (llvm::isa<UnrankedTensorType>(lhsType) ||
      llvm::isa<UnrankedTensorType>(rhsType))
    return emitOpError("unranked tensor operands are not supported");

  bool lhsIsTensor = static_cast<bool>(lhsTensor);
  bool rhsIsTensor = static_cast<bool>(rhsTensor);

  // Scalar-scalar: result must also be scalar.
  if (!lhsIsTensor && !rhsIsTensor)
  {
    if (resultTensor)
      return emitOpError("expected scalar result for scalar lhs and rhs");
  }
  else
  {
    // At least one tensor operand: result must be a tensor whose shape is
    // the broadcast of whichever operand shapes are tensors (a scalar
    // operand imposes no shape constraint of its own).
    if (!resultTensor)
      return emitOpError("expected tensor result when either operand is a tensor");

    if (!resultTensor.hasStaticShape())
      return emitOpError("dynamic result shapes are not yet supported");

    if (lhsIsTensor && !lhsTensor.hasStaticShape())
      return emitOpError("dynamic tensor operand shapes are not yet supported");
    if (rhsIsTensor && !rhsTensor.hasStaticShape())
      return emitOpError("dynamic tensor operand shapes are not yet supported");

    llvm::ArrayRef<int64_t> lhsShape = lhsIsTensor ? lhsTensor.getShape() : llvm::ArrayRef<int64_t>{};
    llvm::ArrayRef<int64_t> rhsShape = rhsIsTensor ? rhsTensor.getShape() : llvm::ArrayRef<int64_t>{};

    if (lhsIsTensor && rhsIsTensor)
    {
      auto broadcastShape = computeBroadcastShape(lhsShape, rhsShape);
      if (mlir::failed(broadcastShape))
        return emitOpError("lhs and rhs tensor shapes are not broadcast-compatible");
      if (*broadcastShape != resultTensor.getShape())
        return emitOpError("result shape does not match the broadcast of "
                           "lhs and rhs shapes");
    }
    else
    {
      // Exactly one tensor operand; the other is a splatted scalar, so
      // result shape must equal the tensor operand's shape exactly.
      llvm::ArrayRef<int64_t> tensorShape = lhsIsTensor ? lhsShape : rhsShape;
      if (tensorShape != resultTensor.getShape())
        return emitOpError("result shape must match the tensor operand's shape");
    }
  }

  // Dtype promotion: float wins if either side is float.
  Type lhsElem = getElementTypeOrSelf(lhsType);
  Type rhsElem = getElementTypeOrSelf(rhsType);
  Type resultElem = getElementTypeOrSelf(resultType);

  bool lhsIsFloat = llvm::isa<FloatType>(lhsElem);
  bool rhsIsFloat = llvm::isa<FloatType>(rhsElem);

  if (lhsIsFloat || rhsIsFloat)
  {
    Type expectedFloat = lhsIsFloat && rhsIsFloat
                             ? (lhsElem.getIntOrFloatBitWidth() >=
                                        rhsElem.getIntOrFloatBitWidth()
                                    ? lhsElem
                                    : rhsElem)
                             : (lhsIsFloat ? lhsElem : rhsElem);
    if (resultElem != expectedFloat)
      return emitOpError("expected float result element type ")
             << expectedFloat << ", got " << resultElem;
  }
  else
  {
    if (!llvm::isa<IntegerType>(resultElem))
      return emitOpError("expected integer result element type when both "
                         "operands are integer");
  }

  return mlir::success();
}
