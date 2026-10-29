# 发布副本验证记录

日期：2026-10-29。以下是源码展示副本的实际检查，不借用其他候选或历史设备运行结果。

## CPU 演示

在 `numba2tilelang/` 内执行 `python -B demo_convert.py` 一次：退出码 0，标准输出 1585 字节、标准错误 0 字节。输出包括 `add` 的 TileLang factory、`T.Parallel` 和尾部边界检查。

仅生成文本，没有导入生成物、调用设备编译或启动 GPU。

## 旧诊断套件：首次运行失败

在相同目录执行：

```sh
python -B -m unittest -v test_convert test_acceptance
```

原始结果：退出码 1，`Ran 22 tests`，`FAILED (failures=7, errors=2)`。失败计数包括 subtest，不能从这些数简单相减推导“通过多少个独立样例”。

| 现象 | 当前解释 |
|---|---|
| `ModuleNotFoundError: No module named 'composition'` | 发布文件清单漏了测试辅助依赖，不是转换器语义缺陷 |
| `A[0, k]` 触发 `selection.py` 内 `KeyError: '#'` | 已观察到内部异常；不是预期的 `Unsupported` |
| 转置 tensor 文本断言失败 | 测试找 `K, M`，实际生成使用 `_tl_K, _tl_M`；至少此断言存在命名口径不一致 |
| 幂、偏移下标、参数标注、非零 range 起点、模块绑定、额外写回的拒绝断言失败 | 测试期望抛出 `Unsupported`，实际未抛；尚未逐项区分能力扩展、过时断言与真实语义问题，不声称这些全部是错误接受 |

不为得到绿色结果删除测试、放松断言或修改冻结基线。首次失败记录保留；补件后的检查单独记录，不覆盖这次失败。

## 补件后复验：当前结果

只补入已审核、原字节的 `composition.py`，再次执行同一 unittest 命令一次：退出码 1，`Ran 22 tests`，`FAILED (failures=7, errors=3)`。没有修改测试或转换器来追求通过。

缺依赖的导入错误已消失。同一用例继续运行后，两个混合映射 subtest 在 `lowering.py:571` 抛出 `Unsupported: 同一张量同一维不能混用不同的轴或系数: A[0]`，与测试预期接受不一致。因此错误计数从 2 变成 3；不意味着补件制造了三个新的实现缺陷。`selection.py` 的 `KeyError` 与七项断言失败仍在。

这组旧测试与当前基线的预期尚未逐项重新对齐，不能作为发布质量全绿的证据；也不能把所有断言失败一概归因于转换语义错误。两轮执行前后副本源码指纹均未改变。

## RVV

发布副本只做静态语法与文件指纹检查。**未在移除 `KMP_DUPLICATE_LIB_OK` workaround 后重新执行 RVV 验证**；没有新的 Clang、QEMU 或硬件数值结论。

## 静态检查

组装时核对前端 183 原始 manifest 固定 SHA 与 29 件源码 SHA；发布 manifest 记录复制前后指纹。扫描发布文本中的明确私人路径、令牌格式、私钥头和内部连接标识；未检出不等于安全审计完备，也不构成版权确权。
