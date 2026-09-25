## Paper

https://arxiv.org/abs/2506.01844

## Citation

```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```

新增 cache_song_pointseg_samples.py (line 1)
离线生成 current point cloud + priors + pseudo labels/weights/scores
输出 manifest.json + shard_*/字段.npy
使用 mmap .npy shard，训练随机读取不会反复解压大文件
默认 float16 存储，减少空间占用
可保存少量 pseudo label PLY 预览

新增 SongPointSegCachedDataset (line 259)
读取离线 cache
返回训练需要的 observation.point_cloud 和 pointseg.* pseudo 字段

更新 train_song_pointseg.py (line 56)
新增 --sample-cache-dir
使用 cache 时跳过在线 generate_pseudo_labels / future point cloud / cdist
模型直接使用缓存的 priors

使用方式：

python src/lerobot/scripts/song_cache_pointseg_samples.py \
  --output-dir /home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/song_pointseg_sample_cache \
  --current-points 50000 \
  --future-points 16384 \
  --batch-size 8 
然后训练：
PYTHONPATH=src conda run -n reap python src/lerobot/scripts/train_song_pointseg.py \
  --sample-cache-dir /path/to/song_pointseg_cache \
  --output-dir /path/to/train_output
  --num-workers 16 \
  --storage-dtype float16 \
  --overwrite
  


  可以，完全不引入目标掩码、抓取/放置阶段识别或夹爪状态判断。只修改时序几何证据的计算方式即可。

最推荐方案是：

## 1. 同一未来帧内比较两个运动假设

当前算法分别跨未来帧取最小值：

```python
held_residual = min_t(d_held[t])
static_residual = min_t(d_static[t])

residual_gap = static_residual - held_residual
```

问题是两个最小值可能来自不同时间，破坏比较意义。

应该保留每个未来帧的距离：

```python
d_held[t]   # 假设点跟随末端
d_static[t] # 假设点静止于世界
```

先逐帧计算：

```python
gap[t] = d_static[t] - d_held[t]
```

含义：

```text
gap > 0：跟随末端假设更合理
gap < 0：世界静止假设更合理
```

## 2. 根据末端运动幅度连续加权

相邻帧运动很小时，两个假设无法区分。这些帧不应和长时间偏移拥有相同权重。

不需要阶段识别，只计算连续运动幅度：

```python
translation_motion = ||future_pose[t].translation||

rotation_motion = rotation_angle(future_pose[t])

motion_baseline = (
    translation_motion
    + object_radius * rotation_motion
)
```

随后得到软权重：

```python
motion_weight = sigmoid(
    (motion_baseline - 0.015) / 0.005
)
```

这样：

- 几乎不动的未来帧权重接近 0；
- 有明显平移或旋转的未来帧权重接近 1；
- 没有任何离散阶段判断。

## 3. 对逐帧证据进行稳健聚合

不建议直接取最大值，容易受错误位姿影响。建议取加权 top-k 平均：

```python
weighted_gap = motion_weight[t] * gap[t]
residual_gap = topk_mean(weighted_gap, k=3)
```

或者使用平滑的 log-sum-exp：

```python
residual_gap = tau * logsumexp(
    weighted_gap / tau,
    dim=future_time,
)
```

更稳妥的版本是“正证据减负证据”：

```python
positive = relu(gap)
negative = relu(-gap)

held_evidence = weighted_mean(positive, motion_weight)
static_evidence = weighted_mean(negative, motion_weight)

residual_gap = held_evidence - 0.5 * static_evidence
```

## 4. 使用相对距离，而非固定毫米差值

当前固定判断：

```python
static_residual - held_residual > 0.006
```

对点云密度和未来采样数很敏感。

可以改成归一化差异：

```python
relative_gap = (
    d_static - d_held
) / (
    d_static + d_held + 0.005
)
```

然后：

```python
motion_score = sigmoid(
    (aggregated_relative_gap - margin) / temperature
)
```

初始可尝试：

```python
margin = 0.10
temperature = 0.10
```

这样 3072、16000、50000 点等不同密度下会更稳定。

## 5. 降低未来点云下采样误差

当前：

```text
current = 50000
future = 16000
```

同一个杯子点可能没有出现在未来下采样结果里，导致 `d_held` 偏大。

在计算伪标签时建议：

```text
current_points = 50000
future_points = 50000
```

如果显存不够，可保持 16000，但对多个未来帧分别采样后进行稳健聚合。不同未来帧采样不同点，本身能提高覆盖率。

## 推荐最终公式

```python
for each future frame t:
    d_held_t = nearest_distance(current_point, future_cloud_t)

    static_query_t = transform_current_point_to_future_frame(
        current_point,
        future_pose_t,
    )
    d_static_t = nearest_distance(static_query_t, future_cloud_t)

    relative_gap_t = (
        d_static_t - d_held_t
    ) / (
        d_static_t + d_held_t + eps
    )

    motion_weight_t = soft_weight_from_pose_displacement(
        future_pose_t
    )

residual_gap = weighted_topk_mean(
    relative_gap_t,
    motion_weight_t,
    k=3,
)

motion_score = sigmoid(
    (residual_gap - margin) / temperature
)
```

整个方法只使用：

- 当前点云；
- 未来点云；
- 连续末端位姿；
- 最近邻几何距离。

不使用任何目标掩码、阶段标签、抓取检测、释放检测或人工语义规则。最关键的修复是：不要再分别对 `held/static` 跨时间取最小值。