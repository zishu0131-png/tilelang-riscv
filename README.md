# TileLang → RISC-V: AI 算子编译研究

面向 RISC-V 的跨架构 AI 算子编译研究原型，包含两个独立组件：**Numba CUDA Python → TileLang 源码转换器**，以及**真实 TileLang/TVM PrimFunc → RVV C 消费器**。

> 这是受限子集的可行性研究，不是生产级编译器，也不是 TileLang 官方注册后端。两个组件的全部能力并未端到端贯通。

## 架构与边界

```text
Numba CUDA Python（源码，静态解析）
  └─ numba2tilelang / 公共 convert
       ├─ CUDA 线程级 TileLang → NVIDIA GPU 路径
       └─ 可移植 tile 级 TileLang
            └─ 真实 tvm.tirx.PrimFunc（默认 CUDA lowering 之前）
                 └─ tilelang2rvv / generate
                      └─ RVV intrinsic C
                           └─ Clang RV64GCV ELF
                                └─ QEMU virt / 单 hart / RVV 1.0
```

- **前端转换器**：静态分析支持范围内的 Numba CUDA kernel，经内部语义表示生成 TileLang。识别受限的逐元素、归约、矩阵乘等模式；不支持的结构应显式拒绝。
- **RVV 消费器**：读取真实 PrimFunc 节点，生成显式 RVV intrinsic C。支持受限连续向量访问、尾部处理与选定 tile 操作，具体接受条件以实现和测试为准。
- **不能混同两条路径**：RVV 消费器拒绝依赖 `threadIdx` 的 SIMT 寻址，不能把任意 CUDA 线程级生成物直接当成可移植 tile IR。
- **功能验证**：RVV 示例配有独立 C 参考与裸机测试程序，可在 VLEN 128/256/512 下运行。QEMU 结果只回答功能问题，不代表真实芯片性能。

## 文件布局

```text
numba2tilelang/   冻结前端基线的源码与自带样例
tilelang2rvv/    IR 消费器、14 个示例、测试、Clang/QEMU runner
docs/           复现说明、已知限制和发布来源
release-manifest.json   发布源码 SHA-256 与来源摘要
```

## 当前验证状态

2026-10-29 发布副本的 CPU 转换演示返回 `rc=0`，仅证明源码转换可运行。随附的旧诊断测试**不是全绿回归套件**；补齐测试依赖后执行 `Ran 22 tests`、`failures=7, errors=3`，存在内部异常与断言/实现行为不一致。首次失败和补件复验均保留，见 [验证记录](docs/validation.md)。RVV 发布副本未重新执行编译或 QEMU 验证。

## 从哪里开始

1. 阅读 [复现指南](docs/reproduction.md)，先做不需要 GPU 的前端转换。
2. 安装兼容 TileLang 与 RVV 工具链，再单独运行 RVV 示例与负控。
3. 阅读 [发布范围与限制](docs/release-scope.md)，区分历史研究记录与此副本实际检查。

本版不包含大规模第三方语料、私有研究账本、构建缓存、GPU 二进制或未合入实验候选。也不发布无法逐件确认来源的外部样例。

## 研究重点

- 让源程序中的索引、边界和计算结构在转换过程中保持可追踪，而不只是生成能解析的代码。
- 将源码接受、生成成功、编译成功、数值正确与性能分别验证。
- 用独立参考和变异负控检验验证程序，避免“打印 PASS”被误当成正确性。
- 将 GPU 线程执行模型与单 hart RVV 执行模型的差异显式化。

## Roadmap

进一步扩展可证明支持的转换子集、前后端串联和数值验证；取得具体芯片接口契约后再研究 VCIX 接入。**当前未实现或验证 VCIX，不宣称 NPU 部署或性能加速。**

## 许可与依赖

此首次展示版尚未选择项目开源许可证，不额外授予源码再分发或再许可权；公开可读不等同于采用 MIT/Apache 等开源许可。后续由项目作者明确选择许可证。

Numba、TileLang/TVM、LLVM/Clang、QEMU 等为独立第三方项目，本仓库不打包这些软件；使用时须遵守各自许可证。本项目不代表上述上游项目，代码和文档的整理使用了 AI 辅助。
