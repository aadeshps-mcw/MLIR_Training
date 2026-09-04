#include "Aadesh/AadeshDialect.h"
#include "Aadesh/AadeshOps.h"

// Define TableGen macro before including pass declarations to instantiate base classes
#define GEN_PASS_DEF_LOWERAADESHTOTOSA
#include "Aadesh/AadeshPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

static bool isUnrankedTensor(Type t) {
  return llvm::isa<UnrankedTensorType>(t);
}

struct ReluOpLowering : public OpConversionPattern<mlir::aadesh::ReluOp> {
  using OpConversionPattern<mlir::aadesh::ReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::aadesh::ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    Type type = input.getType();
    Location loc = op.getLoc();

    // Reject unranked tensors as TOSA requires ranked operands
    if (isUnrankedTensor(type)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower relu on unranked tensor to TOSA; "
              "rank must be known statically");
    }

    Type elemType = getElementTypeOrSelf(type);

    if (auto rankedType = llvm::dyn_cast<RankedTensorType>(type)) {
      if (auto floatType = llvm::dyn_cast<FloatType>(elemType)) {
        FloatAttr minAttr = rewriter.getFloatAttr(floatType, 0.0);
        FloatAttr maxAttr = rewriter.getFloatAttr(
            floatType, std::numeric_limits<double>::infinity());
        rewriter.replaceOpWithNewOp<tosa::ClampOp>(
            op, op.getType(), input, minAttr, maxAttr,
            tosa::NanPropagationMode::PROPAGATE);
        return success();
      }
      if (auto intType = llvm::dyn_cast<IntegerType>(elemType)) {
        unsigned width = intType.getWidth();

        if (intType.isUnsigned()) {
          // relu(x) == x for any unsigned value, clamping to [0, max] would be a no-op anyway, so fold directly.
          rewriter.replaceOp(op, input);
          return success();
        }

        // Signless is treated as signed for TOSA lowering purposes here.
        IntegerAttr minAttr = rewriter.getIntegerAttr(intType, 0);
        IntegerAttr maxAttr = rewriter.getIntegerAttr(
            intType, llvm::APInt::getSignedMaxValue(width));
        rewriter.replaceOpWithNewOp<tosa::ClampOp>(
            op, op.getType(), input, minAttr, maxAttr,
            tosa::NanPropagationMode::PROPAGATE);
        return success();
      }
      return rewriter.notifyMatchFailure(op, "unsupported tensor element type");
    }

    if (auto floatType = llvm::dyn_cast<FloatType>(elemType)) {
      Value zero = arith::ConstantOp::create(
          rewriter, loc, floatType, rewriter.getFloatAttr(floatType, 0.0));
      rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, input, zero);
      return success();
    }
    if (auto intType = llvm::dyn_cast<IntegerType>(elemType)) {
      if (intType.isUnsigned()) {
        // relu(x) == x for any unsigned scalar value.
        rewriter.replaceOp(op, input);
        return success();
      }
      Value zero = arith::ConstantOp::create(
          rewriter, loc, intType, rewriter.getIntegerAttr(intType, 0));
      rewriter.replaceOpWithNewOp<arith::MaxSIOp>(op, input, zero);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "unsupported scalar element type");
  }
};

struct AddOpLowering : public OpConversionPattern<mlir::aadesh::AddOp> {
  using OpConversionPattern<mlir::aadesh::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::aadesh::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Type type = lhs.getType();

    if (isUnrankedTensor(type)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower add on unranked tensor to TOSA; "
              "rank must be known statically");
    }

    Type elemType = getElementTypeOrSelf(type);

    if (llvm::isa<RankedTensorType>(type)) {
      if (llvm::isa<FloatType>(elemType) || llvm::isa<IntegerType>(elemType)) {
        rewriter.replaceOpWithNewOp<tosa::AddOp>(op, op.getType(), lhs, rhs);
        return success();
      }
      return rewriter.notifyMatchFailure(op, "unsupported tensor element type");
    }

    // Scalar path: add is signedness-agnostic for integers (two's complement),
    // so no unsigned special-casing is needed here, unlike relu.
    if (llvm::isa<FloatType>(elemType)) {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, lhs, rhs);
      return success();
    }
    if (llvm::isa<IntegerType>(elemType)) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "unsupported scalar element type");
  }
};

struct MulOpLowering : public OpConversionPattern<mlir::aadesh::MulOp> {
  using OpConversionPattern<mlir::aadesh::MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::aadesh::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Type type = lhs.getType();
    Location loc = op.getLoc();

    if (llvm::isa<UnrankedTensorType>(type)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower mul on unranked tensor to TOSA; "
              "rank must be known statically");
    }

    Type elemType = getElementTypeOrSelf(type);

    if (llvm::isa<RankedTensorType>(type)) {
      // tosa.mul's `shift` must be a rank-1, single-element i8 tensor
      // (TOSA's "scalar tensor" convention is tensor<1xi8>, not tensor<i8>).
      auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
      auto shiftAttr = DenseElementsAttr::get(shiftType, static_cast<int8_t>(0));
      Value shift = tosa::ConstOp::create(rewriter, loc, shiftType, shiftAttr);

      rewriter.replaceOpWithNewOp<tosa::MulOp>(op, op.getType(), lhs, rhs,
                                                shift);
      return success();
    }

    if (llvm::isa<FloatType>(elemType)) {
      rewriter.replaceOpWithNewOp<arith::MulFOp>(op, lhs, rhs);
      return success();
    }
    if (llvm::isa<IntegerType>(elemType)) {
      rewriter.replaceOpWithNewOp<arith::MulIOp>(op, lhs, rhs);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "unsupported scalar element type");
  }
};

struct LowerAadeshToTosa
    : public mlir::aadesh::impl::LowerAadeshToTosaBase<LowerAadeshToTosa> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    target.addIllegalDialect<mlir::aadesh::AadeshDialect>();
    target.addLegalDialect<tosa::TosaDialect, arith::ArithDialect>();

    RewritePatternSet patterns(context);
    patterns.add<ReluOpLowering, AddOpLowering, MulOpLowering>(context);

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::aadesh::createLowerAadeshToTosaPass() {
  return std::make_unique<LowerAadeshToTosa>();
}
