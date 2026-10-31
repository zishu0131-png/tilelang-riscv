# TileLang → RISC-V：AI 算子编译研究

本仓库是总项目入口，介绍研究目标、两段编译链路与版本组合。**两段实现现已分别维护在独立仓库**，本仓库不再复制其源码。

## 项目导航

| 项目 | 职责 | 入口 |
|---|---|---|
| **tilelang-riscv**（本仓库） | 总体架构、研究范围、版本与复现导航 | [版本配套表](docs/versions.md) |
| **numba2tilelang** | Numba CUDA Python → TileLang 源码转换 | [源码与快速开始](https://github.com/zishu0131-png/numba2tilelang) |
| **tilelang2rvv** | 真实 TileLang/TVM PrimFunc → RVV C → RV64GCV ELF | [源码与工具链说明](https://github.com/zishu0131-png/tilelang2rvv) |

三个仓库交叉链接；源码各自只有一个当前权威位置。此处使用普通链接和固定 commit，不要求 Git submodule。

## 架构

```text
Numba CUDA Python（源码静态解析）
  └─ numba2tilelang / convert
       ├─ CUDA 线程级 TileLang → NVIDIA GPU
       └─ 可移植 tile 级子集
            └─ 真实 tvm.tirx.PrimFunc
                 └─ tilelang2rvv / generate
                      └─ RVV intrinsic C
                           └─ Clang → RV64GCV ELF
                                └─ QEMU virt / 单 hart / RVV 1.0
```

**两条路径不能混同**：RVV 消费器拒绝依赖 `threadIdx` 的寻址，不能把任意 CUDA 线程级生成物直接当成可移植 tile IR。两个组件独立可运行，也不等于全部能力已端到端贯通。

## 从哪里开始

1. 在 [numba2tilelang](https://github.com/zishu0131-png/numba2tilelang) 运行无需 GPU 的 `demo_convert.py`，观察源程序如何变成 TileLang 文本。
2. 在 [tilelang2rvv](https://github.com/zishu0131-png/tilelang2rvv) 按说明准备 TileLang、Clang 与 QEMU，复现受限 IR 的 RVV 功能测试。
3. 需要固定版本时使用 [版本配套表](docs/versions.md)，不要把两个仓库的最新 main 自动视为经过整链验证的组合。

更多说明：[复现导航](docs/reproduction.md) · [验证状态](docs/validation.md) · [发布范围](docs/release-scope.md)。

## 当前发布状态

2026-10-31 拆分发布：转换器为 **189 基线的 31 件核心源码**；RVV 为先前公开快照的 8 件源码/支持文件，算法未改。

- 转换器 CPU 演示退出码 0；随附旧诊断套件 `Ran 22 tests`、`failures=7, errors=3`，**不是全绿回归套件**。
- RVV 本次仅做静态与文件身份检查，未重跑 Clang/QEMU。
- 本次没有发布或复验完整 VLM 实验，也不把 GPU 结果外推为 RVV 整链结果。

旧版总仓包含的源码仍可从 [2026-10-29 历史提交](https://github.com/zishu0131-png/tilelang-riscv/tree/3058b8046098b2ff64fab26ba23a0a5f4a40fe8a) 找到；拆仓没有重写或删除 Git 历史。

## 研究原则与路线

追踪索引、布局、边界与执行模型的变化，分别验证源码接受、生成、编译、数值及性能。后续扩展支持子集、固定版本整链复现和独立数值参考，而不是只追求生成代码或打印 PASS。

两段均为受限研究原型，不是生产级编译器或 TileLang 官方后端。QEMU 单 hart 功能结果不代表真实芯片性能、多核部署或 GPU 模型仿真。**VCIX 尚未实现或验证**，需具体芯片契约后另行研究。

## 许可与依赖

三个项目尚未选择额外开源许可证；公开可读不等同于 MIT/Apache 授权。拆仓不改变既有权利状态。Numba、TileLang/TVM、PyTorch、LLVM/Clang、QEMU 为独立第三方项目，须遵守各自许可证，本项目不打包或代表上游软件。代码和文档整理使用了 AI 辅助。
