# YOLO 推理项目 (yolo-inference)

## 项目简介
本项目是用于完成 YOLO 模型推理任务的作业。包含推理代码、依赖说明以及运行文档。

## 环境依赖
- Python 3.8+
- PyTorch
- ultralytics
- 其他依赖请见 `requirements.txt`

## 安装步骤
1. 克隆仓库到本地：
   ```bash
   git clone https://github.com/WANG2026013705/Wang.git
   cd Wang
```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

使用方法

运行推理脚本：

```bash
python inference.py --weights yolov8n.pt --source data/test.jpg
```

目录结构

· inference.py: 推理核心代码
· requirements.txt: 依赖列表
· docs/: 相关文档
