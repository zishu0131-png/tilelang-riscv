# 复现指南

以下命令从仓库根目录执行。先区分**纯 CPU 的源码转换**与**需要 TileLang/交叉工具链的 RVV 功能验证**。

## 1. 前端：无需 GPU 的最小演示

要求 Python 3.10+（源码使用现代类型标注）。普通 tile 级转换及下面两份测试只需标准库和仓库本身。

```sh
python -B numba2tilelang/demo_convert.py
```

目标：终端输出 `add` 对应的 TileLang Python 源码。

这里传入 `convert` 的是源码字符串。**没有执行 Numba kernel，没有导入生成物，也没有编译或启动 GPU**。`from numba import cuda` 是被解析的输入文本，不是本演示进程实际执行的 import。

旧诊断套件（目前存在已知失败，**不是应当全绿的回归承诺**，见 [验证记录](validation.md)）：

```sh
python -B -m unittest discover -s numba2tilelang -p test_convert.py -v
python -B -m unittest discover -s numba2tilelang -p test_acceptance.py -v
```

要弄懂的问题：源程序索引、归约轴和边界分别如何进入生成源码？不支持的输入是否显式失败？结构测试通过与数值测试通过有何差别？

自测：改动一个源程序表达式，观察生成物是否同步变化；不能把“生成的 Python 语法合法”称作数值正确。

### 可选 threads 路径

`thread_typing.py` 的部分路径会延迟导入 Numba；生成物的 `ThreadRunner` 需要 PyTorch。运行生成物还需要兼容 TileLang、CUDA 与相应设备环境。这些**不是上面 CPU 演示的依赖**，本版不提供全套 CUDA 环境锁定或一键 GPU 复现承诺。

## 2. RVV：工具链和依赖

原研究环境记录为 TileLang 0.1.15、Clang 23.1.3、QEMU 11.1.0。这里固定 TileLang 版本，但不声称所有操作系统都有可用 wheel，也不把这些版本之外的组合视为已验证。

```sh
python -m pip install -r requirements-rvv.txt
```

RVV 消费器依赖此 TileLang 绑定的 `tvm.tirx`。它与普通 TVM 的接口及后续 TileLang 版本不能假定兼容。

另安装支持 `riscv64-unknown-elf` / `rv64gcv` 的 Clang 和支持 RVV 的 `qemu-system-riscv64`。仓库不自动下载它们，不修改系统或 Python site-packages。

Windows PowerShell 中指定：

```powershell
$env:RVV_CLANG = "C:\path\to\clang.exe"
$env:RVV_QEMU = "C:\path\to\qemu-system-riscv64.exe"
```

Linux/macOS 的命令形式（**此发布副本尚未跨平台实测**）：

```sh
export RVV_CLANG=/path/to/clang
export RVV_QEMU=/path/to/qemu-system-riscv64
```

未设置时，runner 仍使用 Windows 用户目录下的 Scoop 工具路径回退。不要把这项回退误认为跨平台自动探测。

**OpenMP 提醒：发布副本移除了原版自动设置 `KMP_DUPLICATE_LIB_OK=TRUE` 的 workaround，未在移除后的版本上重新执行 RVV 验证。** 如果导入时出现重复 OpenMP 运行时错误，应清理冲突环境；本仓库不默认忽略它。

## 3. RVV：生成、编译与 QEMU 功能验证

```sh
python -B tilelang2rvv/verify_demo.py
```

该入口对 14 个自带 TileLang 示例读取真实 PrimFunc、生成 RVV C、交叉编译 ELF，再逐个运行 VLEN 128/256/512。输入 IR、生成 C、汇编、ELF 和结果 JSON 写到 `tilelang2rvv/build/`，该目录不进 Git。

预期每次运行打印 `PASS`；这只是复现目标，**不是本发布副本已经跑过的声明**。实际失败时保留输出并报告环境和失败样例。

完整 RVV 测试（会触发 Clang/QEMU，不是纯 Python 测试）：

```sh
python -B -m unittest discover -s tilelang2rvv -p test_backend.py -v
python -B -m unittest discover -s tilelang2rvv -p test_runner.py -v
```

工具缺失时部分测试会 skip；**skip 不计通过**。负控包括错误退出、trap 和生成代码变异；请同时检查测试数量与跳过数量，而不是只看末尾 `OK`。

要弄懂的问题：`T.Parallel` 如何变成动态 VL 循环？尾部怎样裁剪？矩阵 tile 怎样映射为向量微内核？依赖 `threadIdx` 的输入为何拒绝？

动手产出：选择一个示例，对照保存的 `input.tir`、`generated_kernel.c` 和 `kernel.s`，解释索引与边界如何变化；再核对独立参考结果。

自测：明确区分 IR 结构检查、真实交叉编译、QEMU 数值结果和真实硬件性能。最后一项不在当前复现范围。
