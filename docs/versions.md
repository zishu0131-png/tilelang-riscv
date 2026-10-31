# 版本配套表

以下固定的是 **2026-10-31 三仓发布组合**，用于定位源码，不宣称已完成这两个版本的整链兼容性或数值验收。

| 组件 | 发布 commit | 来源 | 本次验证 |
|---|---|---|---|
| [numba2tilelang](https://github.com/zishu0131-png/numba2tilelang) | [`939ba1b994ad3e5b92c6a8d53a64228a5d0f8032`](https://github.com/zishu0131-png/numba2tilelang/tree/939ba1b994ad3e5b92c6a8d53a64228a5d0f8032) | 189 基线 31 件核心原字节，加 5 件辅助/旧诊断/演示 | CPU 演示 rc0；旧诊断 22 tests，7 failures/3 errors |
| [tilelang2rvv](https://github.com/zishu0131-png/tilelang2rvv) | [`dd3edbb74df837df33198fe1512f6c9a9c6504c6`](https://github.com/zishu0131-png/tilelang2rvv/tree/dd3edbb74df837df33198fe1512f6c9a9c6504c6) | 原总仓公开快照的 8 件 RVV 源码/支持文件 | 静态解析/身份检查；未重跑编译与 QEMU |

每个子仓的 `release-manifest.json` 记录源码 SHA-256 与来源。总仓 `components.json` 提供机器可读链接和 commit。

## 固定版本获取

```sh
git clone https://github.com/zishu0131-png/numba2tilelang.git
git -C numba2tilelang checkout 939ba1b994ad3e5b92c6a8d53a64228a5d0f8032
git clone https://github.com/zishu0131-png/tilelang2rvv.git
git -C tilelang2rvv checkout dd3edbb74df837df33198fe1512f6c9a9c6504c6
```

历史总仓快照：[`3058b8046098b2ff64fab26ba23a0a5f4a40fe8a`](https://github.com/zishu0131-png/tilelang-riscv/tree/3058b8046098b2ff64fab26ba23a0a5f4a40fe8a)，包含 183 前端和 RVV 首次公开副本。旧结果与新结果按版本分别引用。
