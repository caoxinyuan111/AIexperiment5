# Multimodal Sentiment Analysis (Experiment 5)

本仓库包含了实验五“多模态情感分类”的完整实现代码。本项目基于 PyTorch 框架，对比了 Late Fusion（后期融合）与 Interactive Fusion（交互式融合）两种架构，并实现了基于 Cross-Attention 的多模态交互机制。

## 1. 实验环境要求 (Requirements)

执行代码前，请确保安装以下依赖库：

- Python 3.10+
- PyTorch (GPU版本推荐)
- Transformers (HuggingFace)
- Torchvision
- Pandas
- Scikit-learn
- Tqdm
- Pillow

你可以使用以下命令一键安装所有依赖（使用清华镜像源）：

pip install torch torchvision transformers pandas scikit-learn tqdm pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

## 2. 代码文件结构

请确保你的文件目录结构如下所示：

StudentID-Name-Exp5/
├── data/                    # 数据集目录
│   ├── data/                # [必须] 原始图片和文本文件 (.jpg, .txt)
│   ├── train.txt            # [必须] 训练集标签文件
│   └── test_without_label.txt # [必须] 测试集索引文件
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── dataset.py           # 数据加载与预处理 (Dataset类)
│   └── model.py             # 模型定义 (含 Late Fusion & Interactive Fusion)
├── main.py                  # 主程序 (包含训练、验证、消融实验、预测全流程)
├── requirements.txt         # 依赖列表
└── README.md                # 项目说明文档

## 3. 执行流程

### 步骤一：准备数据
请确保解压后的原始数据（包含几千个 .txt 和 .jpg 的文件夹）位于 `data/data/` 目录下。并将 `train.txt` 和 `test_without_label.txt` 文件放入 `data/` 目录中。

### 步骤二：运行主程序
在项目根目录下打开终端，执行以下命令：

python main.py

### 步骤三：查看结果
程序运行过程中会自动执行以下操作，无需人工干预：
1.  **模型对比**：自动训练 Late Fusion 和 Interactive Fusion 两种模型，并输出验证集准确率对比表。
2.  **消融实验**：在 Interactive 模型基础上，分别测试 Text-Only（仅文本）和 Image-Only（仅图像）的性能。
3.  **结果生成**：自动选择表现最好的模型权重，对测试集进行预测，并在根目录下生成 `test_result.txt` 文件。

## 4. 参考资料与致谢

本项目的模型设计参考了以下论文和开源仓库：

*   **核心参考论文**: 
    > Xue, P., et al. (2023). "Sentiment classification method based on multitasking and multimodal interactive learning." *Proceedings of CCL 2023*. 
    > (本代码借鉴了该论文中 Cross-Modal Interaction 的核心思想，使用 Cross-Attention 实现模态交互)
    
*   **参考仓库**: 
    > [GloGNN](https://github.com/RecklessRonan/GloGNN) 
    > (参考了其 README 规范与代码结构设计)

*   **基础架构**: 
    *   HuggingFace Transformers (BERT implementation)
    *   PyTorch Torchvision (ResNet implementation)