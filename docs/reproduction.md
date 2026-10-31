# 复现导航

源码已拆成两个独立仓库；本总仓不直接运行转换器或 RVV 测试。

## 1. Numba CUDA → TileLang

```sh
git clone https://github.com/zishu0131-png/numba2tilelang.git
cd numba2tilelang
python -B demo_convert.py
```

要求 Python 3.10+。演示静态解析源码、输出 TileLang 文本，不导入输入 kernel，不需要 GPU。生成成功不等于设备编译或数值正确。

依赖说明及 threads 路径边界见 [转换器 README](https://github.com/zishu0131-png/numba2tilelang#readme)。旧诊断测试存在已知失败，见其 [验证记录](https://github.com/zishu0131-png/numba2tilelang/blob/main/docs/validation.md)。

## 2. TileLang PrimFunc → RVV

```sh
git clone https://github.com/zishu0131-png/tilelang2rvv.git
cd tilelang2rvv
python -m pip install -r requirements.txt
```

按 [RVV README](https://github.com/zishu0131-png/tilelang2rvv#readme) 安装并配置 `RVV_CLANG`、`RVV_QEMU`，再运行 `python -B verify_demo.py`。它会生成 IR/C/汇编/ELF 和运行记录，不是纯 CPU Python 测试。工具缺失的 skip 不计通过。

QEMU 数值验证与真实 RISC-V 硬件性能是不同问题。当前没有 VCIX 契约或验证。

## 3. 两段如何衔接

先检查上游生成的是可移植 tile 子集还是 CUDA 线程寻址；后者不能直接交给拒绝 `threadIdx` 的 RVV 消费器。本次不提供任意输入一键整链转换承诺。

严格定位源码使用 [版本配套表](versions.md) 中的 commit，不只依赖移动的 main 分支。
