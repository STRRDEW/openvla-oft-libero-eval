# 失败分析（真实案例）

> 说明：这里的“现象”来自实际视频帧和逐 episode JSONL；
> “原因猜测”没有经过额外传感器或中间位姿日志验证，不能当成结论。

## 案例 1：基线 k=8，task 9 超时

- suite：`libero_spatial`
- task_id：9
- 指令：`pick up the black bowl on the wooden cabinet and place it on the plate`
- 配置：`num_open_loop_steps=8`，`seed=7`，`num_steps_wait=10`
- episode：`episode_idx=2`，`initial_state_idx=2`
- 初始状态 SHA-256 前 16 位：`6e3698ba2ffc6e03`
- 结果：失败
- 步数：`env_steps=230`，`policy_steps=220`，`model_calls=28`
- 时间：推理 `3.489381s`，仿真 `4.605912s`，墙钟 `12.604976s`
- 视频：`artifacts/phase2/baseline/failure_task9_ep2.mp4`
- 帧：`artifacts/phase2/baseline/t9e2_1s.png`、`t9e2_3s.png`、`t9e2_end.png`

### 观察到的现象

- 1s：黑色碗在右侧木柜上方，机器人夹爪在画面右侧靠近柜子。
- 3s：夹爪移动到柜子前方/上方，但没有看到稳定的抓取闭合。
- 结束帧：黑色碗仍然在木柜上，夹爪空着停在左侧盘子上方。
- 该 episode 跑到 `TASK_MAX_STEPS=220` 的 policy 上限后超时。

### 尚未验证的原因猜测

1. 夹爪接近目标时位置/姿态偏差，抓取闭合时没有夹住碗；
2. 夹住了但抬升/运输阶段滑落，随后策略继续执行“放置”；
3. 策略在柜子附近进入局部来回运动，直到超时。

现有日志只有 episode 级统计和 30fps 视频，没有逐帧末端位姿、夹爪开合、
目标物体位姿，因此不能区分这三类。下一步应记录
`robot0_eef_pos`、`robot0_eef_quat`、`robot0_gripper_qpos` 和每步 action，
并按 approach / grasp / transport / place 分段。

---

## 案例 2：k=1，task 1 超时

- suite：`libero_spatial`
- task_id：1
- 指令：`pick up the black bowl next to the ramekin and place it on the plate`
- 配置：`num_open_loop_steps=1`，`seed=7`，`num_steps_wait=10`
- episode：`episode_idx=7`，`initial_state_idx=7`
- 初始状态 SHA-256 前 16 位：`a45364f84a9e141a`
- 结果：失败
- 步数：`env_steps=230`，`policy_steps=220`，`model_calls=220`，
  `actions_generated=1760`
- 时间：推理 `27.02795s`，仿真 `4.222983s`，视频保存 `0.235142s`，
  墙钟 `35.932685s`；单次推理平均 `122.854ms`
- 视频：`artifacts/phase3/k1/failure_task1_ep7.mp4`
- 帧：`artifacts/phase3/k1/failure_task1_ep7_t1s.png`、`_t3s.png`、`_t5s.png`、`_end.png`

### 观察到的现象

- 1s：目标黑碗在桌面右下方，盘子在中左，机器人夹爪在左侧。
- 3s：夹爪移动到左侧的小金属碗/炉台附近，没有去右下方的目标黑碗。
- 5s：夹爪已经到盘子上方。
- 结束帧：目标黑碗仍在桌面右下方，夹爪空着停在盘子上方。
- 该 episode 每次只执行 1 个动作，共调用模型 220 次，最后超时。

### 尚未验证的原因猜测

1. k=1 时每次只执行动作块的第 1 个动作，模型看到的帧间变化很小，
   可能在高频重规划下产生来回/犹豫；
2. 模型对“目标碗”的定位在初始帧就偏到了另一个碗；
3. 夹爪抓取失败后，策略继续执行放置阶段。

视频只能证明“目标没有被放到盘子里”和“夹爪最后为空”，不能证明是上述
哪一个原因。尤其不能只凭一帧说“模型认错了物体”。

---

## 案例 3：k=1，task 2 超时（补充）

- task_id：2
- 指令：`pick up the black bowl from table center and place it on the plate`
- `episode_idx=9`，`initial_state_idx=9`
- 初始状态 SHA-256 前 16 位：`864819f19088c1c5`
- `env_steps=230`，`policy_steps=220`，`model_calls=220`
- 推理 `27.434429s`，墙钟 `36.946664s`
- 视频：`artifacts/phase3/k1/failure_task2_ep9.mp4`
- 帧：`artifacts/phase3/k1/failure_task2_ep9_t1s.png`、`_t3s.png`、`_t5s.png`、`_end.png`

现象与案例 2 类似：目标碗在桌面中央，夹爪最后在盘子上方，目标仍在桌面。
这支持“k=1 的失败多为超时/未完成放置”，但仍不能确定根因。

---

## 案例 4：k=4，task 8 与 k=8 同一初始状态的分歧

- suite：`libero_spatial`
- task_id：8
- 指令：`pick up the black bowl next to the plate and place it on the plate`
- 初始状态 SHA-256 前 16 位：`ad6e3d768b047e19`
- `episode_idx=4`，`initial_state_idx=4`，`seed=7`
- 三组配置：

| 配置 | 结果 | env_steps | policy_steps | model_calls | 推理 | 墙钟 |
|---|---|---:|---:|---:|---:|---:|
| k=8 | 成功 | 102 | 92 | 12 | 1.502 s | 6.817 s |
| k=4 | 失败 | 230 | 220 | 55 | 6.875 s | 15.761 s |
| k=1 | 失败 | 230 | 220 | 220 | 28.392 s | 37.805 s |

- 视频：`artifacts/phase2/baseline/success_task8_ep4.mp4`、
  `artifacts/phase3/k4/failure_task8_ep4.mp4`、
  `artifacts/phase3/k1/failure_task8_ep4.mp4`
- 帧：`artifacts/phase3/k4/failure_task8_ep4_t1s.png`、`_t3s.png`、
  `_t5s.png`、`_end.png`

### 观察到的现象

- k=8：正常抓取并放置，102 个 env steps 就成功。
- k=4：1s 时两个碗在左侧，3s 夹爪已到盘子上方，5s/结束仍在盘子上方；
  两个碗没有被搬到盘子里。220 个 policy steps 后超时。
- k=1：同样在 220 个 policy steps 后超时。

### 尚未验证的原因猜测

1. k=4 的动作块边界改变了接近/抓取轨迹，导致没有夹住目标；
2. 夹爪闭合时目标位置有偏差，之后策略继续执行“放置”；
3. 目标在运输中滑落，但视频分辨率下无法确认。

这是同一任务、同一初始状态、同一 seed 下最干净的对照，但只有一个
episode；不能据此说“k=4 一定失败”。要定位原因，需要逐 step 记录末端位姿、
夹爪开合、目标物体位姿和 action。

---

## 共同观察与限制

- 所有失败都不是异常崩溃，而是 episode 跑到 **220 个 policy steps** 的上限后超时；
  `error` 字段为空。
- 失败时的 `model_calls` 取决于 k：k=8 是 28 次，k=4 是 55 次，k=1 是 220 次；
  因此不能把“220 次模型调用”当成 k=8 的失败原因。k=1 的 220 次调用只是
  因为它每步调用一次模型。
- 所有失败视频都是 30fps 原始回放，未加速；时长 7.33s，对应 220 帧。
- 当前没有逐帧位姿/夹爪日志，因此“抓取失败”“滑落”“犹豫”都只能是猜测。
- 若要真正定位，应在 runner 中按 step 记录：
  `action`、`robot0_eef_pos`、`robot0_eef_quat`、`robot0_gripper_qpos`、
  `obs` 中目标物体位姿，以及成功判据的中间状态。

## 日志告警（不是失败根因）

- `EGL.eglMakeCurrent ... EGL_NOT_INITIALIZED` 出现在解释器退出阶段；
  进程退出码为 0，视频和成功标志正常。现象已记录，原因未完全验证。
- TensorFlow 在 sm_120 上提示缺少预编译 CUDA kernel；本项目推理未使用
  TensorFlow 计算图，未观察到对结果的影响。
