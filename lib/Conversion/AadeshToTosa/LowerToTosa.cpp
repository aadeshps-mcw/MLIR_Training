#include "Aadesh/AadeshDialect.h"
#include "Aadesh/AadeshOps.h"

#define GEN_PASS_DEF_LOWERAADESHTOTOSA
#include "Aadesh/AadeshPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"

using namespace mlir;

namespace
{

  static bool isUnrankedTensor(Type t) { return llvm::isa<UnrankedTensorType>(t); }

  struct ReluOpLowering : public OpConversionPattern<mlir::aadesh::ReluOp>
  {
    using OpConversionPattern<mlir::aadesh::ReluOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(mlir::aadesh::ReluOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
      Value input = adaptor.getInput();
      Type type = input.getType();
      Location loc = op.getLoc();
      if (isUnrankedTensor(type))
        return rewriter.notifyMatchFailure(op, "cannot lower relu on unranked tensor; rank must be known statically");
      Type elemType = getElementTypeOrSelf(type);
      if (auto rankedType = llvm::dyn_cast<RankedTensorType>(type))
      {
        if (auto floatType = llvm::dyn_cast<FloatType>(elemType))
        {
          FloatAttr minAttr = rewriter.getFloatAttr(floatType, 0.0);
          FloatAttr maxAttr = rewriter.getFloatAttr(floatType, std::numeric_limits<double>::infinity());
          rewriter.replaceOpWithNewOp<tosa::ClampOp>(op, op.getType(), input, minAttr, maxAttr, tosa::NanPropagationMode::PROPAGATE);
          return success();
        }
        if (auto intType = llvm::dyn_cast<IntegerType>(elemType))
        {
          unsigned width = intType.getWidth();
          if (intType.isUnsigned())
          {
            rewriter.replaceOp(op, input);
            return success();
          }
          IntegerAttr minAttr = rewriter.getIntegerAttr(intType, 0);
          IntegerAttr maxAttr = rewriter.getIntegerAttr(intType, llvm::APInt::getSignedMaxValue(width));
          rewriter.replaceOpWithNewOp<tosa::ClampOp>(op, op.getType(), input, minAttr, maxAttr, tosa::NanPropagationMode::PROPAGATE);
          return success();
        }
        return rewriter.notifyMatchFailure(op, "unsupported tensor element type");
      }
      if (auto floatType = llvm::dyn_cast<FloatType>(elemType))
      {
        Value zero = arith::ConstantOp::create(rewriter, loc, floatType, rewriter.getFloatAttr(floatType, 0.0));
        rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, input, zero);
        return success();
      }
      if (auto intType = llvm::dyn_cast<IntegerType>(elemType))
      {
        if (intType.isUnsigned())
        {
          rewriter.replaceOp(op, input);
          return success();
        }
        Value zero = arith::ConstantOp::create(rewriter, loc, intType, rewriter.getIntegerAttr(intType, 0));
        rewriter.replaceOpWithNewOp<arith::MaxSIOp>(op, input, zero);
        return success();
      }
      return rewriter.notifyMatchFailure(op, "unsupported scalar element type");
    }
  };

  struct AddOpLowering : public OpConversionPattern<mlir::aadesh::AddOp>
  {
    using OpConversionPattern<mlir::aadesh::AddOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(mlir::aadesh::AddOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
      Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
      Type type = lhs.getType();
      if (isUnrankedTensor(type))
        return rewriter.notifyMatchFailure(op, "cannot lower add on unranked tensor; rank must be known statically");
      Type elemType = getElementTypeOrSelf(type);
      if (llvm::isa<RankedTensorType>(type))
      {
        if (llvm::isa<FloatType>(elemType) || llvm::isa<IntegerType>(elemType))
        {
          rewriter.replaceOpWithNewOp<tosa::AddOp>(op, op.getType(), lhs, rhs);
          return success();
        }
        return rewriter.notifyMatchFailure(op, "unsupported tensor element type");
      }
      if (llvm::isa<FloatType>(elemType))
      {
        rewriter.replaceOpWithNewOp<arith::AddFOp>(op, lhs, rhs);
        return success();
      }
      if (llvm::isa<IntegerType>(elemType))
      {
        rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
        return success();
      }
      return rewriter.notifyMatchFailure(op, "unsupported scalar element type");
    }
  };

  struct MulOpLowering : public OpConversionPattern<mlir::aadesh::MulOp>
  {
    using OpConversionPattern<mlir::aadesh::MulOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(mlir::aadesh::MulOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
      Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
      Type type = lhs.getType();
      Location loc = op.getLoc();
      if (llvm::isa<UnrankedTensorType>(type))
        return rewriter.notifyMatchFailure(op, "cannot lower mul on unranked tensor; rank must be known statically");
      Type elemType = getElementTypeOrSelf(type);
      if (llvm::isa<RankedTensorType>(type))
      {
        auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
        auto shiftAttr = DenseElementsAttr::get(shiftType, static_cast<int8_t>(0));
        Value shift = tosa::ConstOp::create(rewriter, loc, shiftType, shiftAttr);
        rewriter.replaceOpWithNewOp<tosa::MulOp>(op, op.getType(), lhs, rhs, shift);
        return success();
      }
      if (llvm::isa<FloatType>(elemType))
      {
        rewriter.replaceOpWithNewOp<arith::MulFOp>(op, lhs, rhs);
        return success();
      }
      if (llvm::isa<IntegerType>(elemType))
      {
        rewriter.replaceOpWithNewOp<arith::MulIOp>(op, lhs, rhs);
        return success();
      }
      return rewriter.notifyMatchFailure(op, "unsupported scalar element type");
    }
  };

  // Only float ^ float is currently supported. Integer operands are rejected
  // in PowOpLowering::matchAndRewrite before this is ever reached, so no
  // integer branches are needed here anymore.
  static Value computePowScalar(OpBuilder &b, Location loc, Value base, Value exponent)
  {
    auto baseF = llvm::cast<FloatType>(base.getType());
    auto expF = llvm::cast<FloatType>(exponent.getType());
    if (baseF.getWidth() != expF.getWidth())
    {
      if (baseF.getWidth() < expF.getWidth())
        base = arith::ExtFOp::create(b, loc, expF, base);
      else
        exponent = arith::ExtFOp::create(b, loc, baseF, exponent);
    }
    return math::PowFOp::create(b, loc, base, exponent);
  }
  static AffineMap buildOperandAffineMap(OpBuilder &b, int64_t resultRank, ArrayRef<int64_t> operandShape)
  {
    int64_t operandRank = operandShape.size();
    int64_t offset = resultRank - operandRank;
    SmallVector<AffineExpr> exprs;
    for (int64_t j = 0; j < operandRank; ++j)
    {
      if (operandShape[j] == 1)
        exprs.push_back(b.getAffineConstantExpr(0));
      else
        exprs.push_back(b.getAffineDimExpr(offset + j));
    }
    return AffineMap::get(resultRank, 0, exprs, b.getContext());
  }

  struct PowOpLowering : public OpConversionPattern<mlir::aadesh::PowOp>
  {
    using OpConversionPattern<mlir::aadesh::PowOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(mlir::aadesh::PowOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
      Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
      Location loc = op.getLoc();

      // integer base/exponent support is not implemented yet.
      Type lhsElem = getElementTypeOrSelf(lhs.getType());
      Type rhsElem = getElementTypeOrSelf(rhs.getType());
      if (!llvm::isa<FloatType>(lhsElem) || !llvm::isa<FloatType>(rhsElem))
        return op.emitOpError(
            "aadesh.pow lowering currently only supports float base and "
            "exponent; integer operands are not yet supported");

      bool lhsIsTensor = llvm::isa<TensorType>(lhs.getType());
      bool rhsIsTensor = llvm::isa<TensorType>(rhs.getType());

      if (!lhsIsTensor && !rhsIsTensor)
      {
        rewriter.replaceOp(op, computePowScalar(rewriter, loc, lhs, rhs));
        return success();
      }
      if ((lhsIsTensor && isUnrankedTensor(lhs.getType())) || (rhsIsTensor && isUnrankedTensor(rhs.getType())))
        return rewriter.notifyMatchFailure(op, "cannot lower pow on unranked tensor; rank must be known statically");

      auto resultType = llvm::cast<RankedTensorType>(op.getType());
      if (!resultType.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "dynamic result shapes are not yet supported");

      auto toResultShapedTensor = [&](Value v, bool isTensor) -> Value
      {
        if (isTensor)
          return v;
        auto splatType = RankedTensorType::get(resultType.getShape(), v.getType());
        return tensor::SplatOp::create(rewriter, loc, splatType, v);
      };
      Value lhsTensor = toResultShapedTensor(lhs, lhsIsTensor);
      Value rhsTensor = toResultShapedTensor(rhs, rhsIsTensor);
      auto lhsType = llvm::cast<RankedTensorType>(lhsTensor.getType());
      auto rhsType = llvm::cast<RankedTensorType>(rhsTensor.getType());

      Value init = tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{});
      int64_t rank = resultType.getRank();
      SmallVector<AffineMap, 3> maps = {
          buildOperandAffineMap(rewriter, rank, lhsType.getShape()),
          buildOperandAffineMap(rewriter, rank, rhsType.getShape()),
          rewriter.getMultiDimIdentityMap(rank)};
      SmallVector<utils::IteratorType, 4> iterTypes(rank, utils::IteratorType::parallel);

      auto genericOp = linalg::GenericOp::create(
          rewriter, loc, TypeRange{resultType}, ValueRange{lhsTensor, rhsTensor}, ValueRange{init},
          maps, iterTypes,
          [&](OpBuilder &b, Location bodyLoc, ValueRange args)
          {
            Value result = computePowScalar(b, bodyLoc, args[0], args[1]);
            linalg::YieldOp::create(b, bodyLoc, result);
          });
      rewriter.replaceOp(op, genericOp.getResult(0));
      return success();
    }
  };

  struct ArgMaxOpLowering : public OpConversionPattern<mlir::aadesh::ArgMaxOp>
{
  using OpConversionPattern<mlir::aadesh::ArgMaxOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::aadesh::ArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override
  {
    Value input = adaptor.getInput();
    Location loc = op.getLoc();
    Type inputType = input.getType();

    if (isUnrankedTensor(inputType))
      return rewriter.notifyMatchFailure(op, "cannot lower argmax on unranked tensor; rank must be known statically");

    auto inputTensor = llvm::cast<RankedTensorType>(inputType);
    if (!inputTensor.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic input shapes are not yet supported");

    auto resultTensor = llvm::cast<RankedTensorType>(op.getType());
    int64_t inputRank = inputTensor.getRank();
    ArrayRef<int64_t> inputShape = inputTensor.getShape();
    bool keepDim = op.getKeepDim();

    std::optional<int64_t> normDimOpt;
    if (op.getDim().has_value())
    {
      int64_t d = static_cast<int64_t>(*op.getDim());
      normDimOpt = d < 0 ? d + inputRank : d;
    }

    Value reduceInput = input;
    int32_t axis = 0;
    SmallVector<int64_t> squeezedShape;

    if (normDimOpt.has_value())
    {
      axis = static_cast<int32_t>(*normDimOpt);
      for (int64_t i = 0; i < inputRank; ++i)
        if (i != *normDimOpt)
          squeezedShape.push_back(inputShape[i]);
    }
    else
    {
      int64_t numElements = inputTensor.getNumElements();
      auto flatType = RankedTensorType::get({numElements}, inputTensor.getElementType());
      Value flatShapeVal = tosa::getTosaConstShape(rewriter, loc, ArrayRef<int64_t>{numElements});
      reduceInput = tosa::ReshapeOp::create(rewriter, loc, flatType, input, flatShapeVal);
      axis = 0;
      // squeezedShape stays empty -> tosa.argmax result is rank-0
    }

    Type resultElemType = resultTensor.getElementType();
    auto squeezedType = RankedTensorType::get(squeezedShape, resultElemType);

    auto nanMode = tosa::NanPropagationModeAttr::get(rewriter.getContext(),
                                                     tosa::NanPropagationMode::PROPAGATE);
    Value argmax = tosa::ArgMaxOp::create(
        rewriter, loc, squeezedType, reduceInput,
        rewriter.getI32IntegerAttr(axis), nanMode);

    // Decide the final shape we need to produce:
    //   - dim present + keep_dim=true  -> reinsert the reduced axis as size 1
    //   - dim present + keep_dim=false -> squeezed shape is already final
    //   - dim absent (no-dim case)     -> keep_dim is ignored per torch
    //                                     semantics; result stays rank-0
    //     (this is true whether keep_dim was explicitly false OR true)
    SmallVector<int64_t> finalShape;
    if (normDimOpt.has_value() && keepDim)
    {
      for (int64_t i = 0; i < inputRank; ++i)
        finalShape.push_back(i == *normDimOpt ? 1 : inputShape[i]);
    }
    else
    {
      // Covers: dim present + keep_dim=false, AND the no-dim case.
      finalShape = squeezedShape;
    }

    if (finalShape == squeezedShape)
    {
      rewriter.replaceOp(op, argmax);
      return success();
    }

    Value finalShapeVal = tosa::getTosaConstShape(rewriter, loc, ArrayRef<int64_t>(finalShape));
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, resultTensor, argmax, finalShapeVal);
    return success();
  }
};

  struct LowerAadeshToTosa
      : public mlir::aadesh::impl::LowerAadeshToTosaBase<LowerAadeshToTosa>
  {
    void runOnOperation() override
    {
      MLIRContext *context = &getContext();
      ConversionTarget target(*context);
      target.addIllegalDialect<mlir::aadesh::AadeshDialect>();
      target.addLegalDialect<tosa::TosaDialect, arith::ArithDialect, linalg::LinalgDialect,
                             scf::SCFDialect, math::MathDialect, tensor::TensorDialect,
                             cf::ControlFlowDialect>();
      RewritePatternSet patterns(context);
      patterns.add<ReluOpLowering, AddOpLowering, MulOpLowering, PowOpLowering, ArgMaxOpLowering>(context);
      if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    }
  };

} // namespace

std::unique_ptr<Pass> mlir::aadesh::createLowerAadeshToTosaPass()
{
  return std::make_unique<LowerAadeshToTosa>();
}
