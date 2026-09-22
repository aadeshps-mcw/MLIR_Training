#include "Aadesh/AadeshOps.h"
#include "Aadesh/AadeshDialect.h"

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

::llvm::LogicalResult mlir::aadesh::ApplyMatrixOp::verify() {
  mlir::DenseElementsAttr matrix = getMatrix();
  mlir::ShapedType matrixType = matrix.getType();

  if (matrixType.getRank() != 2)
    return emitOpError("matrix attribute must be a rank-2 (2D) tensor, got rank ")
           << matrixType.getRank();

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// ReluOp::fold()
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult mlir::aadesh::ReluOp::fold(FoldAdaptor adaptor)
{

  // Case 1: The input is a constant.
  if (auto inputAttr = adaptor.getInput())
  {

    // Case 1a: Scalar floating-point constant.
    if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(inputAttr))
    {
      llvm::APFloat value = floatAttr.getValue();

      // Leave NaNs alone so the fold matches arith.maximumf's NaN-propagating
      // semantics.
      if (value.isNegative() && !value.isNaN())
        return mlir::FloatAttr::get(getInput().getType(), 0.0);

      return inputAttr;
    }

    // Case 1b: Scalar integer constant.
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(inputAttr))
    {
      auto intType = mlir::cast<mlir::IntegerType>(getInput().getType());

      // Unsigned integers can never be negative — always a no-op.
      if (intType.isUnsigned())
        return inputAttr;

      llvm::APInt value = intAttr.getValue();
      if (value.isNegative())
        return mlir::IntegerAttr::get(getInput().getType(), 0);

      return inputAttr;
    }

    // Case 1c: Dense (or splat) tensor constant, float element type.
    //
    // DenseElementsAttr::getValues<APFloat>() transparently handles both the
    // splat and non-splat backing storage, so this one path covers both.
    if (auto denseAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(inputAttr))
    {
      auto shapedType = llvm::cast<mlir::ShapedType>(getInput().getType());
      auto elemType = shapedType.getElementType();

      if (llvm::isa<mlir::FloatType>(elemType))
      {
        llvm::SmallVector<llvm::APFloat, 4> results;
        results.reserve(denseAttr.getNumElements());

        for (llvm::APFloat value : denseAttr.getValues<llvm::APFloat>())
        {
          if (value.isNegative() && !value.isNaN())
            results.push_back(llvm::APFloat::getZero(value.getSemantics()));
          else
            results.push_back(value);
        }

        return mlir::DenseElementsAttr::get(shapedType,
                                             llvm::ArrayRef(results));
      }

      // Case 1d: Dense (or splat) tensor constant, integer element type.
      if (auto intElemType = llvm::dyn_cast<mlir::IntegerType>(elemType))
      {
        // Unsigned integer tensors can never be negative — always a no-op.
        if (intElemType.isUnsigned())
          return inputAttr;

        llvm::SmallVector<llvm::APInt, 4> results;
        results.reserve(denseAttr.getNumElements());

        for (llvm::APInt value : denseAttr.getValues<llvm::APInt>())
        {
          if (value.isNegative())
            results.push_back(llvm::APInt::getZero(value.getBitWidth()));
          else
            results.push_back(value);
        }

        return mlir::DenseElementsAttr::get(shapedType,
                                             llvm::ArrayRef(results));
      }
    }
  }

  // Case 2: Unsigned integer input (non-constant), scalar OR tensor.
  //
  // relu(x) == x for unsigned integers because they cannot be negative.
  // getElementTypeOrSelf peels tensor<4xui32> down to ui32.
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(
          mlir::getElementTypeOrSelf(getInput().getType())))
  {
    if (intType.isUnsigned())
      return getInput();
  }

  return {};
}
