"""返回值丢弃的单次原子写契约；不把普通store或同步协议当作exchange。"""
import ast
from semantic import Unsupported
from thread_bindings import Bindings
from thread_ir import TensorArg


def fail(node, message):
    raise Unsupported(f'line {getattr(node, "lineno", 0)}: 原子写子集: {message}')


def is_exchange(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'exch' and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == 'atomic'
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == 'cuda')


def is_atomic_add(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add' and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == 'atomic'
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == 'cuda')


def validate_add(fn, parameters, calls, parents):
    if len(calls) != 1 or any(is_exchange(n) for n in ast.walk(fn)):
        fail(fn, '仅支持一个add调用点且不能混用exch')
    call = calls[0]
    if not isinstance(parents[call], ast.Expr):
        fail(call, 'add返回值必须丢弃，且调用必须是独立语句')
    if call.keywords or len(call.args) != 3:
        fail(call, 'add需要三个位置参数且不接受keywords')
    target, index, value = call.args
    spec = parameters.get(target.id) if isinstance(target, ast.Name) else None
    if not isinstance(spec, TensorArg) or spec.dtype != 'float64' or len(spec.shape) != 1:
        fail(target, 'add目标必须为rank1 float64 TensorArg形参')
    if spec.value_range is not None:
        fail(target, '值域契约参数只读: ' + target.id)
    if not isinstance(value, ast.Constant) or type(value.value) is not int or value.value != 1:
        fail(value, 'add仅支持整数字面量1的计数增量')
    if isinstance(index, ast.Constant) and (type(index.value) is not int
                                           or not 0 <= index.value < spec.shape[0]):
        fail(index, 'add常量下标必须为有效非负整数')
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == target.id and node is not target:
            parent = parents[node]
            if isinstance(parent, ast.Attribute) and parent.value is node:
                if parent.attr == 'size':
                    continue
                use = parents[parent]
                if parent.attr == 'shape' and isinstance(use, ast.Subscript) \
                        and isinstance(use.slice, ast.Constant) and type(use.slice.value) is int \
                        and use.slice.value == 0 and isinstance(use.ctx, ast.Load):
                    continue
            fail(node, 'add目标仅能作为第一实参或读取size/shape[0]，不能别名或混合访问')
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            fail(node, '含atomic的kernel不支持普通张量写入')
    return call


class CudaBindings(Bindings):
    def check(self, node, name):
        if name in self.names:
            fail(node, 'cuda绑定被遮蔽或改写')


def validate_atomic(tree, fn, parameters):
    additions = [n for n in ast.walk(fn) if is_atomic_add(n)]
    if additions:
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        call = validate_add(fn, parameters, additions, parents)
        validate_cuda_binding(tree, fn, call, parents)
        return
    calls = [n for n in ast.walk(fn) if is_exchange(n)]
    if not calls:
        return
    if len(calls) != 1:
        fail(fn, '仅支持一个exch调用点')
    call = calls[0]
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    if not isinstance(parents[call], ast.Expr):
        fail(call, 'exch返回值必须丢弃，且调用必须是独立语句')
    if call.keywords or len(call.args) != 3:
        fail(call, 'exch需要三个位置参数且不接受keywords')
    target, index, _ = call.args
    spec = parameters.get(target.id) if isinstance(target, ast.Name) else None
    if not isinstance(spec, TensorArg) or spec.dtype != 'int32' or len(spec.shape) != 1:
        fail(target, '目标必须为rank1 int32 TensorArg形参')
    if spec.value_range is not None:
        fail(target, '值域契约参数只读: ' + target.id)
    if not isinstance(index, ast.Constant) or type(index.value) is not int \
            or not 0 <= index.value < spec.shape[0]:
        fail(index, '下标必须为有效非负整数常量')
    for node in ast.walk(fn):
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            fail(node, '含atomic的kernel不支持循环')
        if isinstance(node, ast.Name) and node.id == target.id and node is not target:
            fail(node, '目标仅能作为exch第一实参，不能读取、别名或混合写入')
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            fail(node, '含atomic的kernel不支持普通张量写入')
    validate_cuda_binding(tree, fn, call, parents)


def validate_cuda_binding(tree, fn, call, parents):
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)
               and n.level == 0 and n.module == 'numba'
               and any(a.name == 'cuda' and a.asname is None for a in n.names)]
    if not imports:
        fail(call, '需要模块顶层from numba import cuda')
    checker = CudaBindings({'cuda'}, imports)
    for node in tree.body:
        checker.visit(node)
    scope = fn
    while scope is not tree:
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            checker = CudaBindings({'cuda'})
            if hasattr(scope, 'args'):
                checker.visit(scope.args)
            for node in scope.body if isinstance(scope.body, list) else [scope.body]:
                checker.visit(node)
        scope = parents[scope]
