# -*- coding: utf-8 -*-
"""线程数学入口的保守静态绑定检查；不执行模块或解析任意闭包。

只认可模块顶层显式导入。对相关名称的可见重绑定按全作用域拒绝，
不推断执行顺序或分支可达性。动态猴子补丁、别名逃逸后修改模块对象、
exec/globals等外围代码的执行效果不属于此静态源码契约。
"""
import ast
from semantic import Unsupported


def reject(node, name):
    raise Unsupported(f'line {getattr(node, "lineno", 0)}: 数学名称绑定不明确或被遮蔽: {name}')


class Bindings(ast.NodeVisitor):
    """收集一个词法层的绑定；子函数/类只绑定其名字，不混入子层局部量。"""
    def __init__(self, names, allowed_imports=()):
        self.names = set(names)
        self.allowed_imports = set(allowed_imports)

    def check(self, node, name):
        if name in self.names:
            reject(node, name)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.check(node, node.id)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name):
                self.check(node, root.id)
        self.generic_visit(node)

    def visit_arg(self, node):
        self.check(node, node.arg)

    def visit_Import(self, node):
        if node not in self.allowed_imports:
            for alias in node.names:
                self.check(node, alias.asname or alias.name.split('.')[0])

    def visit_ImportFrom(self, node):
        if node not in self.allowed_imports:
            for alias in node.names:
                if alias.name == '*':
                    reject(node, '*')
                self.check(node, alias.asname or alias.name)

    def visit_FunctionDef(self, node):
        self.check(node, node.name)
        # 默认值/装饰器在外层求值；不遍历子函数体。
        for value in (*node.decorator_list, *node.args.defaults,
                      *(v for v in node.args.kw_defaults if v is not None)):
            self.visit(value)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.check(node, node.name)
        for value in (*node.decorator_list, *node.bases, *(k.value for k in node.keywords)):
            self.visit(value)

    def visit_Lambda(self, node):
        for value in (*node.args.defaults, *(v for v in node.args.kw_defaults if v is not None)):
            self.visit(value)

    def visit_Global(self, node):
        for name in node.names:
            self.check(node, name)

    visit_Nonlocal = visit_Global

    def visit_ExceptHandler(self, node):
        self.check(node, node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        self.check(node, node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node):
        self.check(node, node.name)

    def visit_MatchMapping(self, node):
        self.check(node, node.rest)
        self.generic_visit(node)


def validate_math_bindings(tree, fn, modules, functions):
    loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Load)}
    names = loaded & (set(functions) | {'math'})
    if not names:
        return
    if 'math' in names and 'math' not in modules:
        reject(fn, 'math（需要模块顶层 import math）')

    # validate_imports已经证明这些顶层导入来源；其他层导入不充当证明。
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    checker = Bindings(names, imports)
    for node in tree.body:
        checker.visit(node)

    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    scope = fn
    while scope is not tree:
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            checker = Bindings(names)
            if hasattr(scope, 'args'):
                checker.visit(scope.args)
            body = scope.body if isinstance(scope.body, list) else [scope.body]
            for node in body:
                checker.visit(node)
        scope = parents[scope]
