# **MedCoT-7B: 融合思维链技术的轻量级医疗 AI 实践**

**第 32 组课程大作业**：基于 QLoRA 与梯度累加策略，在单卡 11GB 显存受限环境下，实现了 DeepSeek-R1-Distill-Qwen-7B 模型的全量 LoRA 微调，赋予模型医生级的临床辨证思维链（Chain-of-Thought）能力。

## **📖 项目简介 (Introduction)**

本项目旨在解决医疗大模型私有化部署中“高性能”与“低资源”的矛盾。我们选用 **DeepSeek-R1-Distill-Qwen-7B** 作为基座，利用 **medical-o1-reasoning-SFT** 数据集注入专业的医学推理逻辑。

通过引入 **4-bit QLoRA** 量化技术与 **梯度累加 (Gradient Accumulation)** 策略，我们成功打破了 11GB 显存的物理瓶颈，在单张 RTX 2080 Ti 上完成了 3 个 Epoch 的深度微调。实验结果表明，MedCoT-7B 在复杂病例（如外科蝼蛄疖、内科脾虚泄泻）的诊断中，具备了逻辑严密的思维链推理能力，修正了基座模型的标签偏置问题。

## **🌟 核心特性 (Features)**

* **低资源极致优化**：通过 Batch Size=1 \+ Gradient Accumulation=16 策略，在 11GB 显存下实现了等效 Batch Size 16 的训练，无 OOM 溢出。  
* **思维链 (CoT) 对齐**：模型不仅输出诊断结果，还能在 \<think\> 标签内展示完整的病理分析与鉴别诊断过程。  
* **高效训练**：集成 Unsloth 加速框架，训练效率提升约 2.3 倍，总训练时长约 11 小时。  
* **临床逻辑修正**：有效解决了通用模型在长尾病种（如“蝼蛄疖”）上误诊为高频词（如“疳积”）的问题。

## **📂 仓库结构 (Directory Structure)**

.  
├── README.md               \# 项目说明文档  
├── requirements.txt        \# 环境依赖列表  
├── src/                    \# 核心代码  
│   ├── process.py          \# 数据清洗与 CoT 格式化脚本  
│   └── med\_app.py          \# Streamlit 前端演示应用  
├── scripts/                \# 运行脚本  
│   └── run\_train.sh        \# 一键复现训练脚本  
├── data/                   \# 数据集存放目录  
│   ├── dataset\_info.json   \# 数据集注册配置  
│   └── medical\_o1\_sft.json \# (需自行下载) 原始数据集  
└── results/                \# 实验结果与图表  
    ├── final\_loss.png      \# 训练 Loss 收敛曲线  
    ├── final\_ppl.png       \# 困惑度变化曲线  
    ├── final\_lr.png        \# 学习率调度曲线  
    └── comparison/         \# 微调前后病例回答对比图

## **🛠️ 环境安装 (Installation)**

推荐使用 Conda 创建独立环境。

**硬件要求**：

* **GPU**: NVIDIA RTX 2080 Ti (11GB) 或更高配置 (支持 CUDA 11.8+)  
* **RAM**: 16GB+  
* **Disk**: 至少 50GB 可用空间 (用于存放模型权重和数据集)

\# 1\. 创建环境  
conda create \-n medcot python=3.10 \-y  
conda activate medcot

\# 2\. 安装 PyTorch (兼容 CUDA 11.8/12.1)  
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \--index-url \[https://download.pytorch.org/whl/cu121\](https://download.pytorch.org/whl/cu121)

\# 3\. 安装项目依赖 (包含 Unsloth 和 LLaMA-Factory)  
pip install \-r requirements.txt

## **🔬 数据集准备 (Data Preparation)**

1. 下载 **medical-o1-reasoning-SFT** 数据集。  
2. 将数据集文件重命名为 medical\_o1\_sft\_Chinese.json 并放置在 data/ 目录下。  
3. 确保 data/dataset\_info.json 中已注册如下信息：

"medical-o1-reasoning-SFT": {  
  "file\_name": "medical\_o1\_sft\_Chinese.json",  
  "columns": {  
    "prompt": "Question",  
    "query": "",  
    "response": "Response"  
  }  
}

## **🚀 训练复现 (Reproduction)**

我们提供了精确的复现脚本。该配置专为 **11GB 显存** 优化，若显存更大可适当调整 Batch Size。

**运行命令：**

bash scripts/run\_train.sh

**run\_train.sh 的具体内容 (Exact Command)：**

\#\!/bin/bash

\# 开启显存碎片整理，防止 OOM  
export PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True

\# 启动训练  
CUDA\_VISIBLE\_DEVICES=0 llamafactory-cli train \\  
    \--stage sft \\  
    \--do\_train True \\  
    \--model\_name\_or\_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\  
    \--dataset medical-o1-reasoning-SFT \\  
    \--template deepseek \\  
    \--finetuning\_type lora \\  
    \--lora\_target all \\  
    \--output\_dir results/MedCoT-7B-Final \\  
    \--overwrite\_cache \\  
    \--overwrite\_output\_dir \\  
    \--cutoff\_len 2048 \\  
    \--preprocessing\_num\_workers 16 \\  
    \--per\_device\_train\_batch\_size 1 \\  
    \--gradient\_accumulation\_steps 16 \\  
    \--lr\_scheduler\_type cosine \\  
    \--logging\_steps 10 \\  
    \--save\_steps 500 \\  
    \--learning\_rate 5e-5 \\  
    \--num\_train\_epochs 3.0 \\  
    \--quantization\_bit 4 \\  
    \--plot\_loss True \\  
    \--fp16 True \\  
    \--seed 42

* **注**：--seed 42 用于固定随机种子以保证结果可复现。  
* **注**：--per\_device\_train\_batch\_size 1 配合 \--gradient\_accumulation\_steps 16 实现了等效 Batch Size \= 16，是解决 2080 Ti 显存溢出的关键。

## **📊 实验结果 (Results)**

模型在 3 个 Epoch 后达到收敛，具体指标如下：

| 指标 (Metrics) | 数值 (Value) | 说明 |  
| Training Loss | 1.4623 | 损失函数平稳下降，表明模型充分拟合思维链数据。 |  
| Perplexity (PPL) | 4.31 | 困惑度显著降低，对医疗术语的预测更精准。 |  
| Training Time | \~11h | 单卡 2080 Ti 高效完成。 |

### **训练曲线图**

*(请在 results/ 目录下查看详细大图)*

## **🩺 推理与演示 (Inference & Demo)**

训练完成后，可以使用 Streamlit 启动带有思维链展示的 Web 界面：

\# 1\. 启动 API 后端 (加载微调后的 Adapter)  
CUDA\_VISIBLE\_DEVICES=0 API\_PORT=8000 llamafactory-cli api \\  
    \--model\_name\_or\_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\  
    \--adapter\_name\_or\_path results/MedCoT-7B-Final \\  
    \--template deepseek \\  
    \--finetuning\_type lora \\  
    \--quantization\_bit 4

\# 2\. 启动前端页面 (另开终端)  
streamlit run src/med\_app.py

### **典型案例对比**

**输入**：1岁幼儿，夏季头皮出现多处小结节，溃破流脓，皮下有空洞。

* **微调前**：误诊为“疳积”或“痄腮”，逻辑混乱。  
* **微调后**：\<think\> 标签内准确识别“夏季湿热”、“皮下空洞”特征，**确诊为“蝼蛄疖”**。

## **📝 引用与致谢 (Citation)**

本项目基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 和 [Unsloth](https://github.com/unslothai/unsloth) 构建。感谢 FreedomIntelligence 提供的开源医疗数据集。

*Created by Group 32 for the Deep Learning Course Project.*