#include "Aadesh/AadeshDialect.h"
#include "Aadesh/AadeshOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/DialectConversion.h"

#include <functional>

#define GEN_PASS_DEF_LOWERAADESHPOWTOLOOPS
#define GEN_PASS_DEF_LOWERAADESHARGMAXTOLOOPS
#include "Aadesh/AadeshPasses.h"

using namespace mlir;

namespace
{

//===----------------------------------------------------------------------===//
// PowOp lowering
//===----------------------------------------------------------------------===//

  static Value expBySquaring(OpBuilder &b, Location loc, Value base, Value exp)
  {
    auto expTy = llvm::cast<IntegerType>(exp.getType());
    unsigned bits = expTy.getWidth();
    bool isFloatBase = llvm::isa<FloatType>(base.getType());
    bool expMaybeNeg = expTy.isSigned() || expTy.isSignless();

    Value zero = arith::ConstantOp::create(b, loc, expTy, b.getIntegerAttr(expTy, 0));

    if (expMaybeNeg)
    {
      Value isNeg = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, exp, zero);

      if (isFloatBase)
      {
        auto fTy = llvm::cast<FloatType>(base.getType());
        Value oneF = arith::ConstantOp::create(b, loc, fTy, b.getFloatAttr(fTy, 1.0));
        Value recip = arith::DivFOp::create(b, loc, oneF, base);
        base = arith::SelectOp::create(b, loc, isNeg, recip, base);

        Value negExp = arith::SubIOp::create(b, loc, zero, exp);
        exp = arith::SelectOp::create(b, loc, isNeg, negExp, exp);
      }
      else
      {
        Value nonNeg = arith::XOrIOp::create(
            b, loc, isNeg,
            arith::ConstantOp::create(b, loc, b.getIntegerAttr(b.getI1Type(), 1)));
        cf::AssertOp::create(b, loc, nonNeg, "aadesh.pow: negative exponent not supported for integer base");
      }
    }

  Value accInit = isFloatBase
                      ? (Value)arith::ConstantOp::create(b, loc, base.getType(), b.getFloatAttr(llvm::cast<FloatType>(base.getType()), 1.0))
                      : (Value)arith::ConstantOp::create(b, loc, base.getType(), b.getIntegerAttr(llvm::cast<IntegerType>(base.getType()), 1));

  Value lb = arith::ConstantIndexOp::create(b, loc, 0);
  Value ub = arith::ConstantIndexOp::create(b, loc, bits);
  Value step = arith::ConstantIndexOp::create(b, loc, 1);

  auto loop = scf::ForOp::create(
      b, loc, lb, ub, step, ValueRange{accInit, base, exp},
      [&](OpBuilder &nb, Location nloc, Value iv, ValueRange iterArgs)
      {
        Value acc = iterArgs[0], curBase = iterArgs[1], curExp = iterArgs[2];
        Value one = arith::ConstantOp::create(nb, nloc, expTy, nb.getIntegerAttr(expTy, 1));
        Value bit = arith::AndIOp::create(nb, nloc, curExp, one);
        Value zeroI = arith::ConstantOp::create(nb, nloc, expTy, nb.getIntegerAttr(expTy, 0));
        Value bitSet = arith::CmpIOp::create(nb, nloc, arith::CmpIPredicate::ne, bit, zeroI);

        Value mulAcc = isFloatBase ? (Value)arith::MulFOp::create(nb, nloc, acc, curBase)
                                   : (Value)arith::MulIOp::create(nb, nloc, acc, curBase);
        Value newAcc = arith::SelectOp::create(nb, nloc, bitSet, mulAcc, acc);
        Value newBase = isFloatBase ? (Value)arith::MulFOp::create(nb, nloc, curBase, curBase)
                                    : (Value)arith::MulIOp::create(nb, nloc, curBase, curBase);
        Value oneShift = arith::ConstantOp::create(nb, nloc, expTy, nb.getIntegerAttr(expTy, 1));
        Value newExp = arith::ShRUIOp::create(nb, nloc, curExp, oneShift);
        scf::YieldOp::create(nb, nloc, ValueRange{newAcc, newBase, newExp});
      });
  return loop.getResult(0);
}

static AffineMap getBroadcastIndexingMap(OpBuilder &b,
                                         ArrayRef<int64_t> operandShape,
                                         ArrayRef<int64_t> resultShape)
{
  int64_t resultRank = resultShape.size();
  int64_t operandRank = operandShape.size();
  int64_t rankDiff = resultRank - operandRank;
  SmallVector<AffineExpr, 4> exprs;
  exprs.reserve(operandRank);
  for (int64_t i = 0; i < operandRank; ++i)
  {
    int64_t resultDim = i + rankDiff;
    if (operandShape[i] == 1 && resultShape[resultDim] != 1)
      exprs.push_back(b.getAffineConstantExpr(0));
    else
      exprs.push_back(b.getAffineDimExpr(resultDim));
  }
  return AffineMap::get(resultRank, /*symbolCount=*/0, exprs, b.getContext());
}

struct PowOpLowering : public OpConversionPattern<mlir::aadesh::PowOp>
{
  using OpConversionPattern<mlir::aadesh::PowOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(mlir::aadesh::PowOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override
  {
    Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
    Location loc = op.getLoc();

    if (!llvm::isa<IntegerType>(getElementTypeOrSelf(rhs.getType())))
      return op.emitOpError("exp-by-squaring lowering requires an integer exponent");

    bool lhsIsTensor = llvm::isa<TensorType>(lhs.getType());
    bool rhsIsTensor = llvm::isa<TensorType>(rhs.getType());

    if (!lhsIsTensor && !rhsIsTensor)
    {
      rewriter.replaceOp(op, expBySquaring(rewriter, loc, lhs, rhs));
      return success();
    }

    if ((lhsIsTensor && llvm::isa<UnrankedTensorType>(lhs.getType())) ||
        (rhsIsTensor && llvm::isa<UnrankedTensorType>(rhs.getType())))
      return rewriter.notifyMatchFailure(op, "unranked tensors not supported");

    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    if (!resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not yet supported");

    if (lhsIsTensor && !llvm::cast<RankedTensorType>(lhs.getType()).hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic tensor operand shapes not yet supported");
    if (rhsIsTensor && !llvm::cast<RankedTensorType>(rhs.getType()).hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic tensor operand shapes not yet supported");

    ArrayRef<int64_t> resultShape = resultType.getShape();
    ArrayRef<int64_t> lhsShape = lhsIsTensor
                                     ? llvm::cast<RankedTensorType>(lhs.getType()).getShape()
                                     : ArrayRef<int64_t>{};
    ArrayRef<int64_t> rhsShape = rhsIsTensor
                                     ? llvm::cast<RankedTensorType>(rhs.getType()).getShape()
                                     : ArrayRef<int64_t>{};
    int64_t rank = resultType.getRank();

    AffineMap lhsMap = getBroadcastIndexingMap(rewriter, lhsShape, resultShape);
    AffineMap rhsMap = getBroadcastIndexingMap(rewriter, rhsShape, resultShape);
    AffineMap outMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap, 3> maps = {lhsMap, rhsMap, outMap};
    SmallVector<utils::IteratorType, 4> iterTypes(rank, utils::IteratorType::parallel);

    Value init = tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{});
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{lhs, rhs}, ValueRange{init},
        maps, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args)
        {
          Value result = expBySquaring(b, bodyLoc, args[0], args[1]);
          linalg::YieldOp::create(b, bodyLoc, result);
        });
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct LowerAadeshPowToLoops
    : public mlir::aadesh::impl::LowerAadeshPowToLoopsBase<LowerAadeshPowToLoops>
{
  void runOnOperation() override
  {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addIllegalOp<mlir::aadesh::PowOp>();
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, linalg::LinalgDialect,
                           tensor::TensorDialect, cf::ControlFlowDialect>();
    RewritePatternSet patterns(context);
    patterns.add<PowOpLowering>(context);
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// ArgMaxOp lowering
//===----------------------------------------------------------------------===//

// One scf.for reduction loop over `ub` elements; extracts each candidate
// via `extractAt(iv)`, keeps first occurrence on ties (strict `>` only).
static std::pair<Value, Value>
buildArgMaxReduction(OpBuilder &b, Location loc, Type elemType,
                     Type indexElemType, Value lb, Value ub, Value step,
                     Value initVal,
                     llvm::function_ref<Value(OpBuilder &, Location, Value)> extractAt)
{
  bool isFloat = llvm::isa<FloatType>(elemType);
  Value initIdx = arith::ConstantOp::create(b, loc, indexElemType,
                                            b.getIntegerAttr(indexElemType, 0));

  auto loop = scf::ForOp::create(
      b, loc, lb, ub, step, ValueRange{initVal, initIdx},
      [&](OpBuilder &nb, Location nloc, Value iv, ValueRange iterArgs)
      {
        Value curVal = iterArgs[0];
        Value curIdx = iterArgs[1];
        Value elem = extractAt(nb, nloc, iv);

        Value isGreater =
            isFloat
                ? (Value)arith::CmpFOp::create(nb, nloc, arith::CmpFPredicate::OGT, elem, curVal)
                : (Value)arith::CmpIOp::create(nb, nloc, arith::CmpIPredicate::sgt, elem, curVal);

        Value ivIdx = arith::IndexCastOp::create(nb, nloc, indexElemType, iv);
        Value newVal = arith::SelectOp::create(nb, nloc, isGreater, elem, curVal);
        Value newIdx = arith::SelectOp::create(nb, nloc, isGreater, ivIdx, curIdx);
        scf::YieldOp::create(nb, nloc, ValueRange{newVal, newIdx});
      });

  return {loop.getResult(0), loop.getResult(1)};
}

struct ArgMaxOpLowering : public OpConversionPattern<mlir::aadesh::ArgMaxOp>
{
  using OpConversionPattern<mlir::aadesh::ArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::aadesh::ArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override
  {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    if (llvm::isa<UnrankedTensorType>(input.getType()))
      return rewriter.notifyMatchFailure(op, "unranked tensors not supported");

    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());

    if (!inputType.hasStaticShape() || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not yet supported");

    Type inputElemType = inputType.getElementType();
    Type indexElemType = resultType.getElementType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::outs() <<"Input shape: ";
    for (int64_t dim : inputShape)
      llvm::outs() << dim << " ";
    llvm::outs() << "\n";
    int64_t inputRank = inputType.getRank();

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    std::optional<int64_t> dimAttr = op.getDim();

    // Case 1: dim absent -- flatten and reduce over every element.
    if (!dimAttr.has_value())
    {
      int64_t numElements = inputType.getNumElements();

      Value flatInput = input;
      if (inputRank != 1)
      {
        ReassociationIndices allDims;
        for (int64_t i = 0; i < inputRank; ++i)
          allDims.push_back(i);
        auto flatType = RankedTensorType::get({numElements}, inputElemType);
        flatInput = tensor::CollapseShapeOp::create(
            rewriter, loc, flatType, input,
            ArrayRef<ReassociationIndices>{allDims});
      }

      Value ub = arith::ConstantIndexOp::create(rewriter, loc, numElements);
      Value initVal = tensor::ExtractOp::create(rewriter, loc, flatInput, ValueRange{c0});

      auto [bestVal, bestIdx] = buildArgMaxReduction(
          rewriter, loc, inputElemType, indexElemType, c1, ub, c1, initVal,
          [&](OpBuilder &b, Location l, Value iv) -> Value {
            return tensor::ExtractOp::create(b, l, flatInput, ValueRange{iv});
          });
      (void)bestVal;

      Value initTensor = tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{});
      Value finalResult =
          tensor::InsertOp::create(rewriter, loc, bestIdx, initTensor, ValueRange{});
      rewriter.replaceOp(op, finalResult);
      
      return success();
    }

    // Case 2: dim present -- reduce along one axis; keep_dim controls
    // whether that axis survives (size 1) in the result.
    int64_t dim = *dimAttr;
    int64_t normDim = dim < 0 ? dim + inputRank : dim;
    bool keepDim = op.getKeepDim();

    SmallVector<int64_t> outerDims;
    for (int64_t i = 0; i < inputRank; ++i)
      if (i != normDim)
        outerDims.push_back(i);

    Value reducedUb = arith::ConstantIndexOp::create(rewriter, loc, inputShape[normDim]);
    Value resultInit = tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{});

    SmallVector<Value> outerIvs;
    std::function<Value(OpBuilder &, Location, size_t, Value)> buildOuter;
    buildOuter = [&](OpBuilder &b, Location l, size_t depth, Value tensorArg) -> Value
    {
      if (depth == outerDims.size())
      {
        SmallVector<Value> baseIndices(inputRank);
        for (size_t k = 0; k < outerDims.size(); ++k)
          baseIndices[outerDims[k]] = outerIvs[k];

        baseIndices[normDim] = c0;
        Value initVal = tensor::ExtractOp::create(b, l, input, baseIndices);

        auto [bestVal, bestIdx] = buildArgMaxReduction(
            b, l, inputElemType, indexElemType, c1, reducedUb, c1, initVal,
            [&](OpBuilder &ib, Location il, Value iv) -> Value {
              SmallVector<Value> idxs = baseIndices;
              idxs[normDim] = iv;
              return tensor::ExtractOp::create(ib, il, input, idxs);
            });
        (void)bestVal;

        SmallVector<Value> outIndices;
        for (int64_t i = 0; i < inputRank; ++i)
        {
          if (i == normDim)
          {
            if (keepDim)
              outIndices.push_back(c0);
          }
          else
          {
            size_t k = 0;
            while (outerDims[k] != i)
              ++k;
            outIndices.push_back(outerIvs[k]);
          }
        }

        return tensor::InsertOp::create(b, l, bestIdx, tensorArg, outIndices);
      }

      int64_t d = outerDims[depth];
      Value ub = arith::ConstantIndexOp::create(b, l, inputShape[d]);
      auto loop = scf::ForOp::create(
          b, l, c0, ub, c1, ValueRange{tensorArg},
          [&](OpBuilder &nb, Location nl, Value iv, ValueRange iterArgs)
          {
            outerIvs.push_back(iv);
            Value updated = buildOuter(nb, nl, depth + 1, iterArgs[0]);
            outerIvs.pop_back();
            scf::YieldOp::create(nb, nl, ValueRange{updated});
          });
      return loop.getResult(0);
    };

    Value finalResult = buildOuter(rewriter, loc, 0, resultInit);
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

struct LowerAadeshArgMaxToLoops
    : public mlir::aadesh::impl::LowerAadeshArgMaxToLoopsBase<LowerAadeshArgMaxToLoops>
{
  void runOnOperation() override
  {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addIllegalOp<mlir::aadesh::ArgMaxOp>();
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect>();
    RewritePatternSet patterns(context);
    patterns.add<ArgMaxOpLowering>(context);
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::aadesh::createLowerAadeshPowToLoopsPass()
{
  return std::make_unique<LowerAadeshPowToLoops>();
}

std::unique_ptr<Pass> mlir::aadesh::createLowerAadeshArgMaxToLoopsPass()
{
  return std::make_unique<LowerAadeshArgMaxToLoops>();
}
