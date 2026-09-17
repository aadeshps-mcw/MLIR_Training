from mlir_aadesh.ir import Context, Module, InsertionPoint, F32Type, Location
from mlir_aadesh.dialects import aadesh, arith


def build_module():
    with Context() as ctx, Location.unknown():
        aadesh.register_dialect(ctx)

        f32 = F32Type.get()
        module = Module.create()
        with InsertionPoint(module.body):
            lhs = arith.constant(f32, 2.0)
            rhs = arith.constant(f32, 3.0)
            aadesh.AddOp(lhs, rhs)

        return module


def main():
    module = build_module()
    print(module)
    assert module.operation.verify()
    print("Module verified OK")


if __name__ == "__main__":
    main()
