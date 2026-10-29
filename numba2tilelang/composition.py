# -*- coding: utf-8 -*-
"""显式源码计算图：每个节点经convert生成，运行器只校验张量及调用kernel。"""
import ast
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from convert import convert, parse_source, detect_pattern, Unsupported


@dataclass(frozen=True)
class Node:
    source: str
    bindings: dict
    shape: tuple
    extents: tuple = ()
    gemm_policy: str = 'scalar'


def dimension(text, shape, extents):
    # layout长度仅允许整数或两种结构化维度引用；不eval生成器文本。
    node = ast.parse(text, mode='eval').body
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
            and node.value.id in ('_tl_shape', '_tl_extents') \
            and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
        values = shape if node.value.id == '_tl_shape' else extents
        index = node.slice.value
        if 0 <= index < len(values):
            return values[index]
    raise Unsupported('不能解析已验证布局维度: ' + text)


def describe(node):
    spec = detect_pattern(parse_source(node.source))
    if node.gemm_policy not in ('scalar', 'tf32'):
        raise ValueError('invalid gemm_policy')
    if len(node.shape) != len(spec.store_index) or any(type(d) is not int or d <= 0 for d in node.shape + node.extents):
        raise ValueError('节点shape/extents必须是正确rank的正整数')
    if spec.program is not None:
        layout = spec.layout
        if len(node.extents) != layout['extra']:
            raise ValueError('独立归约维数量不匹配')
        resolve = lambda text: dimension(text, node.shape, node.extents)
        dims = {name: tuple(resolve(d) for d in values) for name, values in layout['dims'].items()}
        for text, expected in layout['checks']:
            if resolve(text) != expected:
                raise ValueError('源常量归约维不匹配')
        for left, right in layout.get('equalities', ()):
            if resolve(left) != resolve(right):
                raise ValueError('源码要求对应维度长度相等')
        reduction = tuple(resolve(text) for text in layout['extents'])
    else:
        if node.extents:
            raise ValueError('逐元素节点不能有独立归约维')
        dims = {name: tuple(node.shape[spec.store_index.index(axis)] for axis in axes)
                for name, axes in spec.read_axes.items()}
        dims[spec.store_target] = node.shape
        reduction = ()
    if spec.pattern == 'gemm' and node.gemm_policy == 'tf32':
        args, kwargs = [*node.shape, reduction[0]], {}
    elif spec.pattern == 'gemm' or spec.pattern.startswith('reduce_'):
        args, kwargs = [list(node.shape), reduction[0]], {}
    elif spec.pattern in ('pipeline', 'nested'):
        args, kwargs = [list(node.shape)], {'reduction_extents': list(node.extents)}
    else:
        args, kwargs = [list(node.shape)], {}
    return spec, dims, args, kwargs


def graph_manifest(inputs, nodes, outputs):
    tensors = {}
    for name, shape in inputs.items():
        shape = tuple(shape)
        if not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError('图输入shape必须为正整数')
        tensors[name] = list(shape)
    records = []
    for number, node in enumerate(nodes):
        spec, dims, args, kwargs = describe(node)
        if set(node.bindings) != set(spec.params):
            raise ValueError('节点绑定必须完整对应源形参')
        target = node.bindings[spec.store_target]
        if target in tensors:
            raise ValueError('重复生产者或原地输出: ' + target)
        reads = []
        for name in spec.params:
            if name == spec.store_target:
                continue
            tensor = node.bindings[name]
            if tensor not in tensors:
                raise ValueError('输入尚未定义/前向引用: ' + tensor)
            if tuple(tensors[tensor]) != dims[name]:
                raise ValueError('节点输入shape不匹配: ' + tensor)
            reads.append(tensor)
        tensors[target] = list(dims[spec.store_target])
        records.append({'file': f'node_{number}.py', 'function': spec.name,
                        'inputs': reads, 'output': target, 'args': args, 'kwargs': kwargs,
                        'pattern': spec.pattern, 'gemm_policy': node.gemm_policy})
    if not outputs or any(name not in tensors for name in outputs):
        raise ValueError('图输出未定义')
    return {'inputs': list(inputs), 'outputs': list(outputs), 'shapes': tensors, 'nodes': records}


def export_graph(inputs, nodes, outputs, directory):
    """仅导出到新目录；先验证整个图，原CUDA源码不会被exec。"""
    manifest = graph_manifest(inputs, nodes, outputs)
    generated = [convert(node.source, gemm_policy=node.gemm_policy) for node in nodes]
    directory = Path(directory)
    directory.mkdir(parents=False, exist_ok=False)
    for index, (node, code) in enumerate(zip(nodes, generated)):
        (directory / f'node_{index}.py').write_text(code, encoding='utf-8')
        (directory / f'node_{index}_source.py').write_text(node.source, encoding='utf-8')
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (directory / 'graph.py').write_text(
        'from pathlib import Path\nfrom composition import load_graph\n'
        'build = lambda: load_graph(Path(__file__).parent)\n', encoding='utf-8')
    return manifest


def load_graph(directory):
    """只加载本工具生成且可信的模块；不得用于不可信目录。"""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    kernels = []
    for index, node in enumerate(manifest['nodes']):
        spec = importlib.util.spec_from_file_location(f'generated_node_{index}', directory / node['file'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # JSON数组回读为list；TileLang在进入工厂前就按参数做哈希缓存。
        args = [tuple(value) if isinstance(value, list) else value for value in node['args']]
        kwargs = {key: tuple(value) if isinstance(value, list) else value
                  for key, value in node['kwargs'].items()}
        kernels.append(getattr(module, node['function'])(*args, **kwargs))

    def run(inputs, *, return_intermediates=False):
        if set(inputs) != set(manifest['inputs']):
            raise ValueError('图输入名不匹配')
        values = dict(inputs)
        device = None
        for name, value in values.items():
            if list(value.shape) != manifest['shapes'][name] or str(value.dtype) != 'torch.float32':
                raise ValueError('输入shape/dtype不匹配: ' + name)
            if not value.is_cuda or not value.is_contiguous():
                raise ValueError('输入必须是连续CUDA张量: ' + name)
            if device is not None and device != value.device:
                raise ValueError('所有图输入必须在同一设备')
            device = value.device
        for node, kernel in zip(manifest['nodes'], kernels):
            result = kernel(*(values[name] for name in node['inputs']))
            if list(result.shape) != manifest['shapes'][node['output']] \
                    or str(result.dtype) != 'torch.float32' or result.device != device:
                raise RuntimeError('生成kernel输出违反图契约')
            values[node['output']] = result
        return values if return_intermediates else {name: values[name] for name in manifest['outputs']}
    return run
