# 验证状态

## 2026-10-31 独立仓库发布

| 对象 | 实际检查 | 不代表什么 |
|---|---|---|
| 转换器 189 的 31 件核心与 5 件辅助/演示/旧诊断 | 文件 SHA、Python AST、本地 import 闭包；CPU 演示 rc0；旧诊断 `Ran 22 tests` / 7 failures / 3 errors | 不代表完整回归、GPU/VLM 数值或性能通过 |
| RVV 的 8 件既有公开文件 | 保持旧公开字节、SHA、Python AST、本地 import 闭包 | 本次未跑 Clang/QEMU 或真实硬件 |
| 三仓发布关系 | README 交叉链接、组件固定 commit、GitHub 默认分支与提交身份 | 不代表两段整链兼容性认证 |

详细范围随源码维护：
- [numba2tilelang 验证记录](https://github.com/zishu0131-png/numba2tilelang/blob/main/docs/validation.md)
- [tilelang2rvv 验证记录](https://github.com/zishu0131-png/tilelang2rvv/blob/main/docs/validation.md)

旧诊断套件未为发布而删除或放宽断言，未把所有失败笼统归因于转换缺陷。RVV 副本去掉默认 OpenMP 绕过后的数值复验仍未完成。研究环境里的其他局部结果不自动转借到此次发布。

## 历史记录

2026-10-29 首次总仓快照使用转换器 183 基线：CPU演示 rc0，诊断首轮 7F2E，补辅助依赖后 7F3E；RVV 发布副本未重跑。原说明保留在 [历史版本验证文档](https://github.com/zishu0131-png/tilelang-riscv/blob/3058b8046098b2ff64fab26ba23a0a5f4a40fe8a/docs/validation.md)。历史文档不覆盖当前独立仓库结果。
