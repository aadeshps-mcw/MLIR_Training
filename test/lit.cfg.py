import lit.formats

config.name = "AADESH"
config.test_format = lit.formats.ShTest(False)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.aadesh_obj_root, 'test')

config.environment['PATH'] = os.path.pathsep.join((
    config.aadesh_tools_dir, config.llvm_tools_dir, config.environment['PATH']))
