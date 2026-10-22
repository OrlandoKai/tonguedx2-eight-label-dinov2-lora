# TongueDx2 八标签复现：DINOv2 ViT-S/14 + LoRA

本目录实现一个可复现实验支路，使用 release 中 `seg/` 下已经裁剪并分割的舌图，训练八个独立 sigmoid 标签。原始 `origin/`、`seg/` 和 `list/` 数据不修改。

## 论文配置与本实现

| 项目 | 论文报告 | 本实现 |
|---|---|---|
| Backbone | DINOv2 ViT-S/14，21M 参数 | Hugging Face Transformers `facebook/dinov2-small`，同为 ViT-S/14 |
| LoRA | rank 16、alpha 32；适配 qkv、fc1、fc2 | rank 16、alpha 32；映射到 HF 的 query/key/value、mlp.fc1/fc2（不含 attention output projection） |
| 输入 | 336×336 | 336×336，直接读取 release 的 `seg/` 舌体裁剪图 |
| Readout | GAP → Linear(512) → ReLU → Dropout(0.3) → 分类头 | GAP(不含 CLS) → Linear(512) → ReLU → Dropout(0.3) → 8 logits |
| 优化 | AdamW、AMP、cosine warm restarts、EMA=0.999；论文未报告 LoRA 的精确 LR/weight decay/batch size | AdamW + AMP + cosine warm restarts + EMA=0.999；LoRA LR 1e-4、head LR 3e-4、weight decay 0.01、batch 4、梯度累积 4；未报告项是本复现的显式默认值，不冒称论文原值 |
| Loss | 论文对其标签组使用 BCE+正类权重；其 DINOv2 比较在 976 张专家标注图像、11 组/38 类设置上 | 八个二元标签使用 BCEWithLogitsLoss，`pos_weight=N_neg/N_pos`，只从当前训练 fold 计算 |
| 训练 | 论文最多 80 epoch；图像级五折 CV | 每个 release fold 最多 80 epoch；validation macro-AUPRC 早停/选 checkpoint；validation 上逐标签选择 F1 阈值，固定后评估测试集 |
| 增强 | restrained color jitter；避免强 hue/color mixing | 水平翻转、±5°旋转、轻度颜色增强（brightness/saturation 0.05、hue 0.01）；没有 MixUp/CutMix |

**复现边界：**论文报告的 DINOv2+LoRA 得分不是这套八标签 release split 的结果。论文 v17 使用 976 张专家标注图像和不同的 11 组分类结构；因此此目录是按论文参数迁移到 release 八标签任务的独立复现实验，结果不能标注为原论文数值复现。论文没有公开训练代码；DINOv2 backbone 从公开 checkpoint 下载。

## 论文与来源

1. *What Makes Deep Learning Work for Traditional Chinese Medicine Tongue Diagnosis? A Comprehensive Ablation Study*（[arXiv:2607.28148](https://arxiv.org/abs/2607.28148)，预印本）：报告 DINOv2 ViT-S/14 + LoRA v17，输入 336、r=16、alpha=32、qkv/fc1/fc2 适配及 512 维 readout；其 976 张专家标注子集的 DINOv2 结果为 weighted-F1 0.6501±0.0099。数据划分、标签结构与本目录的八标签任务不同，只用于配置参考。
2. Chen et al., *Dr. Tongue: Sign-Oriented Multi-label Detection for Remote Tongue Diagnosis*, [AAAI 2025](https://doi.org/10.1609/aaai.v39i2.32230)：在 TongueDx 的八项舌象属性上研究 SignNet；论文报告 SignNet 平均 F1 75.83%。它是更贴近八标签任务的结构化识别对照，但并非本目录 release fold 上的结果。
3. DINOv2 权重：[Hugging Face `facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small)；模型定义与权重来自公开 DINOv2。
4. CUDA wheel 配对：[PyTorch 官方 previous versions](https://pytorch.org/get-started/previous-versions/) 页面给出 Windows CUDA 12.8 的 PyTorch 2.7.1 / torchvision 0.22.1 安装组合。

**论文未报告且本实现显式设定的值：**LoRA dropout=0.05、LoRA/head 学习率=1e-4/3e-4、weight decay=0.01、batch=4、梯度累积=4、AdamW 的其余默认项、warm-restart 的 T0=10/Tmult=2、80 epoch/早停 patience=12，以及验证 AUPRC 选模策略。它们属于复现方案，不应写成原论文原值。

## 文件

- `config.json`：固定的模型、优化与数据配置。
- `download_model.py`：将官方 DINOv2-S/14 Transformers checkpoint 下载到 `models/dinov2-small/`。
- `train.py`：按 fold 训练、验证阈值校准、固定 test 评估；保存 LoRA/head 权重和逐标签指标。
- `requirements.txt`：CUDA PyTorch 需按本机 CUDA/驱动版本单独安装；其余依赖见文件。

## 运行

本机验证环境为 Python 3.11、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、Transformers 4.57.6。RTX 4060 使用 CUDA 12.8 wheel；驱动已提供 CUDA 12.9，兼容运行 CUDA 12.8 wheel。数据集需从获准的 TongueDx2 release 单独取得，不包含在代码仓库中。克隆项目后在 PowerShell 中设置 `$releaseRoot` 指向本地 release 目录：

```powershell
cd "path/to/dinov2_vits14_lora"
$releaseRoot = "D:/path/to/TongueDx2/release"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe train.py --fold 1 --workers 0 --release-root $releaseRoot
```

在当前受限 Windows 环境中 DataLoader 多进程被系统拒绝，因此使用 `--workers 0`。完成 fold 1 后，依次运行 `--fold 2` 至 `--fold 5`。每折结果写入 `runs/foldN/`。默认路径按本目录的上级 release 布置；也可用 `--release-root` 和 `--model-dir` 覆盖。

先安装与本机 CUDA 驱动匹配的 CUDA PyTorch 和 torchvision，再安装本目录依赖。4060 8GB 建议从 batch 4、梯度累积 4 起跑。脚本启用 AMP；若仍 OOM，使用 `--batch-size 2 --grad-accum 8`。训练启动时会打印实际设备及显存；不会自动退回 CPU 训练。

## 指标与数据边界

checkpoint 按 validation macro-AUPRC 选择；每个标签的 F1 阈值仅由 validation 选取；test 只在模型和阈值锁定后评估。脚本先检查 train/val/test 的 subject ID 和 image path 交集。输出保存 aggregate/per-label 指标，不导出样本级预测。
