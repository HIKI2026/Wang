import argparse
import cv2
import os
import time
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO装甲板视频推理脚本")
    parser.add_argument('--model', type=str, default='best.pt', help='模型权重路径')
    parser.add_argument('--source', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='输出视频保存路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"[INFO] 正在加载模型: {args.model}")
    model = YOLO(args.model)
    
    print(f"[INFO] 正在读取视频: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件: {args.source}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"[INFO] 开始推理... 总帧数: {total_frames}")

    frame_count = 0
    start_time = time.time() # 开始计时

    # 注意这里的 while 是顶格写的！
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[INFO] 处理进度: {frame_count}/{total_frames}")

        results = model.predict(frame, conf=args.conf, device=args.device, verbose=False)
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes
            keypoints = result.keypoints
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id] 
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if keypoints is not None and len(keypoints) > i:
                    kpts = keypoints.xy[i].cpu().numpy()
                    for kp in kpts:
                        kx, ky = int(kp[0]), int(kp[1])
                        if kx > 0 and ky > 0: 
                            cv2.circle(frame, (kx, ky), 5, (0, 0, 255), -1) 
                            cv2.putText(frame, str(i+1), (kx+5, ky-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    end_time = time.time() # 结束计时
    print(f"[INFO] 推理完成！结果已保存至: {args.output}")
    print(f"[INFO] 整个视频推理总耗时: {end_time - start_time:.2f} 秒") # 打印耗时

if __name__ == "__main__":
    main()