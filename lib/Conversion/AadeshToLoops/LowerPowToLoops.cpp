#include "Aadesh/AadeshDialect.h"
#include "Aadesh/AadeshOps.h"

#define GEN_PASS_DEF_LOWERAADESHPOWTOLOOPS
#include "Aadesh/AadeshPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace
{

  static Value expBySquaring(OpBuilder &b, Location loc, Value base, Value exp)
  {
    auto expTy = llvm::cast<IntegerType>(exp.getType());
    unsigned bits = expTy.getWidth();
    bool isFloatBase = llvm::isa<FloatType>(base.getType());

    if (expTy.isSigned() || expTy.isSignless())
    {
      Value zero = arith::ConstantOp::create(b, loc, expTy, b.getIntegerAttr(expTy, 0));
      Value nonNeg = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, exp, zero);
      cf::AssertOp::create(b, loc, nonNeg, "aadesh.pow: negative exponent not supported");
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

} // namespace

std::unique_ptr<Pass> mlir::aadesh::createLowerAadeshPowToLoopsPass()
{
  return std::make_unique<LowerAadeshPowToLoops>();
}