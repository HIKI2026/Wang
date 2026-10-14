import argparse
import cv2
import os
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO装甲板视频推理脚本")
    # 按要求：模型路径、视频路径和输出位置应通过参数或明确的配置修改
    parser.add_argument('--model', type=str, default='best.pt', help='模型权重路径 (如 best.pt)')
    parser.add_argument('--source', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='输出视频保存路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备，如 cpu 或 0 (代表GPU)')
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 加载模型权重
    print(f"[INFO] 正在加载模型: {args.model}")
    if not os.path.exists(args.model):
        print(f"[ERROR] 找不到模型文件: {args.model}")
        return
    model = YOLO(args.model)
    
    # 2. 读取视频
    print(f"[INFO] 正在读取视频: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {args.source}")
        return

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置视频写入器 (保存推理结果视频)
    # 常用的编码器有 'mp4v' 或 'XVID'。如果保存的视频无法播放，可以尝试换成 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"[INFO] 开始推理... 视频分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[INFO] 处理进度: {frame_count}/{total_frames}")

        # 3. 执行推理
        # 允许使用CPU或GPU，这里根据args.device决定
        results = model.predict(frame, conf=args.conf, device=args.device, verbose=False)
        
        # 获取推理结果（针对当前帧）
        result = results[0]
        
        # 4. 绘制检测结果
        # 绘制检测框和类别、置信度 (result.plot() 可以自动完成，但为了自定义关键点，我们手动绘制或用plot()后再加)
        # 这里我们手动提取并根据要求绘制，确保关键点来自模型
        if result.boxes is not None:
            boxes = result.boxes
            # 提取关键点
            keypoints = result.keypoints
            
            for i, box in enumerate(boxes):
                # 提取坐标、类别、置信度
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]  # 类别名称
                
                # 绘制检测框 (绿色)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签和置信度
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 绘制关键点（要求：必须来自模型预测结果，不能使用检测框角点，且清晰可见）
                if keypoints is not None and len(keypoints) > i:
                    # 获取当前目标的4个关键点坐标 (x, y)
                    # keypoints.xy 是 shape=(num_boxes, num_keypoints, 2)
                    kpts = keypoints.xy[i].cpu().numpy()
                    
                    # 绘制关键点（用红色圆点，半径5，确保清晰可见）
                    for kp in kpts:
                        kx, ky = int(kp[0]), int(kp[1])
                        if kx > 0 and ky > 0: # 过滤掉无效点
                            cv2.circle(frame, (kx, ky), 5, (0, 0, 255), -1) # -1 表示实心圆
                            # 或者为了区分不同目标的点，可以使用颜色循环
                            # 为了让点更加显眼，还可以在点旁边加上小数字或连接线
                            cv2.putText(frame, str(i+1), (kx+5, ky-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 写入结果视频
        out.write(frame)

    # 5. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 推理完成！结果已保存至: {args.output}")

if __name__ == "__main__":
    main()