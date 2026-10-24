# 实验结果

所有数字来自真实推理评测，原始逐 episode 记录在
`results/phase2/baseline/episodes.jsonl`、`results/phase3/k1/episodes.jsonl`、
`results/phase3/k4/episodes.jsonl`。本文的汇总由 `tools/analyze_results.py`
从这些 JSONL 重新计算。

## 1. 实验协议

- 模型：`moojink/openvla-7b-oft-finetuned-libero-spatial`（官方权重，未训练）
- 仿真：LIBERO `libero_spatial`，10 个任务 × 每任务 10 个 episode
- 固定：seed=7、初始状态列表、`num_steps_wait=10`、`env_img_res=256`、
  `center_crop=True`
- 唯一变量：每个动作块执行前 k 个动作后重新观察；k = 1、4、8
- 最大步数：220 个策略步；另外前 10 步 dummy action 用于稳定物体
- 计时：`get_action` 前后调用 `torch.cuda.synchronize()`；墙钟包含仿真、
  图像处理、视频保存和 Python 开销

> k=8 复用阶段 2 基线；k=1、k=4 是阶段 3 新跑的。三者使用相同的任务、
> 初始状态和 seed，但运行时间不同。k=4 与 k=1 是独立的完整 100 episode 评测。

## 2. 主结果

| k | episodes | successes | success rate | 95% Wilson CI | mean model calls / episode | mean latency / call | mean wall / episode | inference share of wall |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 8（基线） | 100 | 99 | 99.0% | 94.6%–99.8% | 13.32 | 126.28 ms | 7.93 s | 21.2% |
| 4 | 100 | 99 | 99.0% | 94.6%–99.8% | 26.51 | 125.92 ms | 9.51 s | 35.1% |
| 1 | 100 | 94 | 94.0% | 87.5%–97.2% | 111.76 | 124.43 ms | 20.28 s | 68.6% |

解释：

- k=4 与 k=8 的总体成功率相同（都是 99/100），但 k=4 的平均模型调用约
  为 k=8 的 **2.0 倍**，平均墙钟约高 **20%**。
- k=1 的平均调用约为 k=8 的 **8.4 倍**，平均墙钟约为 **2.6 倍**，
  成功率低 5 个百分点。
- 单次推理延迟在三种配置下都在 124–126 ms，差别很小；主要成本来自
  **调用次数**，不是单次推理变慢。
- 99/100 与 94/100 的 Wilson 区间有重叠，不能宣称 k=1 显著更差；
  这是 100 episode 的小规模评测，不是论文复现。

## 3. 逐任务成功率

| task | k=8 | k=4 | k=1 |
|---:|---:|---:|---:|
| 0 | 10/10 | 10/10 | 10/10 |
| 1 | 10/10 | 10/10 | 9/10 |
| 2 | 10/10 | 10/10 | 9/10 |
| 3 | 10/10 | 10/10 | 10/10 |
| 4 | 10/10 | 10/10 | 10/10 |
| 5 | 10/10 | 10/10 | 9/10 |
| 6 | 10/10 | 10/10 | 10/10 |
| 7 | 10/10 | 10/10 | 9/10 |
| 8 | 10/10 | 9/10 | 9/10 |
| 9 | 9/10 | 10/10 | 9/10 |
| **总计** | **99/100** | **99/100** | **94/100** |

## 4. 计算开销

| k | total model calls | total inference | total simulation | total wall | calls / success | inference s / success |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1,332 | 168.04 s | 240.84 s | 793.30 s | 13.45 | 1.70 s |
| 4 | 2,651 | 333.73 s | 231.85 s | 951.41 s | 26.78 | 3.37 s |
| 1 | 11,176 | 1,390.87 s | 245.54 s | 2,027.79 s | 118.89 | 14.80 s |

> 总墙钟是 run 的实际耗时；逐 episode 的 `wall_seconds` 汇总后与总墙钟一致
> （四舍五入）。它包含视频保存、图像处理和 Python 开销，不只是 GPU 推理。

## 5. 一个受控的失败对照

同一个 task 8、同一个初始状态 `ad6e3d768b047e19`、同一个 seed=7：

| 配置 | 结果 | env steps | policy steps | model calls | inference | wall |
|---|---|---:|---:|---:|---:|---:|
| k=8（基线） | 成功 | 102 | 92 | 12 | 1.502 s | 6.817 s |
| k=4 | 失败/超时 | 230 | 220 | 55 | 6.875 s | 15.761 s |
| k=1 | 失败/超时 | 230 | 220 | 220 | 28.392 s | 37.805 s |

这是本实验里最干净的一组对照：任务、初始状态、seed、模型和预处理都相同，
只改变了执行长度。它说明“在这个初始状态下”，k=8 的成功轨迹在 k=4/k=1
的高频重规划下没有被复现；但单个 episode 不能证明 k=4 一定导致失败，也不能
区分是抓取失败、动作边界不一致，还是超时前没有完成放置。

视频：`artifacts/phase2/baseline/success_task8_ep4.mp4`、
`artifacts/phase3/k4/failure_task8_ep4.mp4`、
`artifacts/phase3/k1/failure_task8_ep4.mp4`。

## 6. 失败模式

- k=8 的唯一失败：task 9 ep2，220 policy steps 后超时；结束帧目标碗仍在
  木柜上，夹爪空着停在盘子上方。
- k=4 的唯一失败：task 8 ep4，同一初始状态在 k=8 下成功；结束帧两个碗仍在
  左侧，夹爪空着停在盘子上方。
- k=1 的 6 个失败：全部跑到 220 个 policy steps 上限后超时（k=1 每步调用一次，
  所以也是 220 次模型调用），分布在 task 1、2、5、7、8、9；
  其中 task 8 ep4 与上面的受控对照是同一个初始状态。
- 这些现象只能说明“没有在预算内完成放置并超时”；具体是没抓到、滑落、
  还是动作选择错误，当前日志没有腕部力/接触/物体位姿，不能确认。

## 7. 原始文件

- 逐 episode：三个 `episodes.jsonl`
- 汇总：`results/analysis/all/comparison.csv`、`per_task.csv`、`failures.csv`
- 图表：`results/analysis/all/success_by_task.png`、`compute_by_k.png`
- 本地副本：`artifacts/analysis/all/`
- 视频：`artifacts/phase2/baseline/`、`artifacts/phase3/k1/`、`artifacts/phase3/k4/`
