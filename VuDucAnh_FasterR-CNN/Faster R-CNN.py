import os
import cv2
import torch
import torchvision
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import math


import seaborn as sns
from sklearn.metrics import confusion_matrix
# ===============================================

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# =======================================================
# 1  Dataset VOC cho Faster R-CNN (Không đổi)
# =======================================================
class VOCDataset(Dataset):
    def __init__(self, img_dirs, ann_dir, transforms=None):
        self.ann_dir = ann_dir
        self.transforms = transforms
        
        # 1. Định nghĩa TẤT CẢ các lớp (class) hợp lệ
        self.valid_classes = {
            "non_defective_phone", 
            "defective", 
            "non-phone"  
        }
        
        self.class_map = {
            "non_defective_phone": 1,
            "defective": 2,
            "non-phone": 3  
        }

        annots = sorted([f for f in os.listdir(ann_dir) if f.lower().endswith('.xml')])
        
        self.image_map = {} 
        for dir_path in img_dirs:
            for img_name in os.listdir(dir_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    base_name = os.path.splitext(img_name)[0]
                    self.image_map[base_name] = os.path.join(dir_path, img_name)

        self.samples = []
        print("[INFO] Đang quét, kiểm tra class và toạ độ BBox...")
        
        # 2. Lọc XML ngay khi khởi tạo
        for ann_name in annots:
            base = os.path.splitext(ann_name)[0]
            if base in self.image_map:
                img_path = self.image_map[base]
                ann_path = os.path.join(ann_dir, ann_name)
                
                try:
                    img_temp = cv2.imread(img_path)
                    if img_temp is None:
                        print(f"[CẢNH BÁO] Không đọc được ảnh: {img_path}, bỏ qua.")
                        continue
                    h, w, _ = img_temp.shape
                    
                    tree = ET.parse(ann_path)
                    root = tree.getroot()
                    
                    valid_object_count = 0
                    for obj in root.findall("object"):
                        
                        name = obj.find("name").text.strip() 
                        
                        # A. Kiểm tra Class
                        if name in self.valid_classes: 
                            bbox = obj.find("bndbox")
                            
                            # B. Kiểm tra và lọc toạ độ BBox
                            xmin = int(float(bbox.find("xmin").text))
                            ymin = int(float(bbox.find("ymin").text))
                            xmax = int(float(bbox.find("xmax").text))
                            ymax = int(float(bbox.find("ymax").text))

                            xmin = max(0, xmin)
                            ymin = max(0, ymin)
                            xmax = min(w, xmax)
                            ymax = min(h, ymax)

                            # C. Kiểm tra kích thước
                            if xmin < xmax and ymin < ymax:
                                valid_object_count += 1
                            else:
                                print(f"[CẢNH BÁO] Bỏ qua vật thể có BBox không hợp lệ trong tệp: {ann_name}")
                                
                    if valid_object_count > 0:
                        self.samples.append((img_path, ann_path))
                    else:
                        print(f"[CẢNH BÁO] Bỏ qua tệp: {ann_name} (không có vật thể hợp lệ NÀO)")
                        
                except Exception as e:
                    print(f"[LỖI] Bỏ qua tệp {ann_name} do lỗi: {e}")

        print(f"[INFO] Đã lọc xong. Tổng cộng {len(self.samples)} ảnh hợp lệ để huấn luyện.")


    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape 

        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes, labels = [], []
        
        for obj in root.findall("object"):
            
            name = obj.find("name").text.strip() 
            
            if name in self.class_map:
                label_to_add = self.class_map[name]
                
                bbox = obj.find("bndbox")
                
                xmin = max(0, int(float(bbox.find("xmin").text)))
                ymin = max(0, int(float(bbox.find("ymin").text)))
                xmax = min(w, int(float(bbox.find("xmax").text)))
                ymax = min(h, int(float(bbox.find("ymax").text)))

                if xmin < xmax and ymin < ymax:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label_to_add)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img = T.ToTensor()(img)
        target = {"boxes": boxes, "labels": labels}
        
        return img, target

    def __len__(self):
        return len(self.samples)


# =======================================================
# 2  Huấn luyện Faster R-CNN
# =======================================================
def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(boxA_area + boxB_area - inter + 1e-6)


def plot_confusion_matrix(cm, class_names):
    """
    Vẽ ma trận nhầm lẫn (cm) bằng Seaborn.
    class_names: danh sách tên các lớp (ví dụ: ['Background', 'ClassA', 'ClassB'])
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    print("[INFO] Đã tạo hình ảnh ma trận nhầm lẫn.")

def evaluate_model(model, test_dataset, device, iou_thresh=0.7, score_thresh=0.8):
    """
    Đánh giá model.
    Trả về:
    - acc (float): Độ chính xác theo cách tính của bạn (TP / Total GT)
    - y_true_cm (list): Danh sách TẤT CẢ các nhãn thật (để vẽ CM)
    - y_pred_cm (list): Danh sách TẤT CẢ các nhãn dự đoán (để vẽ CM)
    """
    model.eval()
    
    # 1. Biến cho Accuracy 
    correct_acc = 0
    total_gt = 0
    
    # 2. Biến cho Confusion Matrix 
    # Class 0 là 'Background'
    y_true_cm = []
    y_pred_cm = []
    
    with torch.no_grad():
        # Lặp qua từng ảnh trong test_dataset 
        for i in range(len(test_dataset)):
            img, target = test_dataset[i] # Lấy ảnh và nhãn
            img = img.to(device)
            outputs = model([img])[0] # Chuyển [img] để tạo batch size 1
            
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            pred_boxes = outputs["boxes"].cpu().numpy()
            pred_scores = outputs["scores"].cpu().numpy()
            pred_labels = outputs["labels"].cpu().numpy()

            # Tính Accuracy  
            total_gt += len(gt_boxes)
            gt_matched_for_acc = [False] * len(gt_boxes)

            for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
                if ps < score_thresh: # Ngưỡng tin cậy (0.8)
                    continue
                
                best_iou = -1
                best_gt_idx = -1

                # Tìm ground truth box khớp nhất
                for g_idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    # Nếu chưa khớp và ĐÚNG nhãn
                    if not gt_matched_for_acc[g_idx] and gl == pl:
                        iou = box_iou(pb, gb)
                        if iou >= iou_thresh and iou > best_iou: # Ngưỡng IoU (0.7)
                            best_iou = iou
                            best_gt_idx = g_idx
                
                # Nếu tìm thấy một cặp khớp
                if best_gt_idx != -1:
                    correct_acc += 1
                    gt_matched_for_acc[best_gt_idx] = True # Đánh dấu là đã khớp

            # B. Chuẩn bị dữ liệu cho Confusion Matrix
            
            # Lọc các dự đoán theo ngưỡng tin cậy
            valid_mask = pred_scores >= score_thresh
            pred_boxes_filt = pred_boxes[valid_mask]
            pred_labels_filt = pred_labels[valid_mask]
            
            gt_matched_cm = [False] * len(gt_boxes)
            pred_matched_cm = [False] * len(pred_boxes_filt)

            # Tìm các cặp khớp 1-1 (greedy theo IoU tốt nhất)
            for p_idx in range(len(pred_boxes_filt)):
                pb = pred_boxes_filt[p_idx]
                pl = pred_labels_filt[p_idx]
                
                best_iou = -1
                best_gt_idx = -1
                
                # Tìm GT có IoU cao nhất với prediction này (bất kể class)
                for g_idx in range(len(gt_boxes)):
                    if gt_matched_cm[g_idx]: # Bỏ qua GT đã khớp
                        continue
                    iou = box_iou(pb, gt_boxes[g_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                
                # Nếu IoU vượt ngưỡng
                if best_iou >= iou_thresh: # Dùng ngưỡng IoU (0.7)
                    gt_matched_cm[best_gt_idx] = True
                    pred_matched_cm[p_idx] = True
                    
                    # Đây là True Positive (nếu class đúng) 
                    # hoặc Classification Error (nếu class sai)
                    y_true_cm.append(gt_labels[best_gt_idx])
                    y_pred_cm.append(pl)
                
                # (Nếu best_iou < iou_thresh, nó sẽ được xử lý ở bước FP)
                
            # Xử lý False Positives (các dự đoán không khớp với GT nào)
            for p_idx in range(len(pred_boxes_filt)):
                if not pred_matched_cm[p_idx]:
                    y_true_cm.append(0) # Nhãn thật là Background
                    y_pred_cm.append(pred_labels_filt[p_idx])
            
            # Xử lý False Negatives (các GT không được dự đoán)
            for g_idx in range(len(gt_boxes)):
                if not gt_matched_cm[g_idx]:
                    y_true_cm.append(gt_labels[g_idx]) # Nhãn thật
                    y_pred_cm.append(0) # Dự đoán là Background
                    
    # Tính toán accuracy cuối cùng
    acc = 100.0 * correct_acc / max(1, total_gt)
    
    # Trả về cả acc và 2 danh sách cho CM
    return acc, y_true_cm, y_pred_cm


#  Hàm train_faster_rcnn 
def train_faster_rcnn(train_loader, test_dataset, epochs=10):
    print("[INFO] === Bắt đầu huấn luyện Faster R-CNN ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dùng thiết bị: {device}")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 4 lớp: 0(bg), 1(non_defective), 2(defective), 3(NON-PHONE)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4) 
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    loss_history, acc_history = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Lặp (iterate) qua train_loader một cách bình thường
        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Đánh giá trên tập test
    
        acc, _, _ = evaluate_model(model, test_dataset, device, iou_thresh=0.7, score_thresh=0.8) 
        
        loss_history.append(avg_loss)
        acc_history.append(acc)
       
        print(f"\n[INFO] Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Test Accuracy: {acc:.2f}%")

    #  Vẽ Ma trận nhầm lẫn 
    print("\n[INFO] Huấn luyện hoàn tất. Đánh giá cuối cùng và vẽ ma trận nhầm lẫn...")
    
    # 1. Đánh giá lần cuối để lấy y_true_all, y_pred_all
    final_acc, y_true_all, y_pred_all = evaluate_model(model, test_dataset, device, iou_thresh=0.7, score_thresh=0.8)
    print(f"[INFO] Final Test Accuracy (theo logic của bạn): {final_acc:.2f}%")

    # 2. Lấy tên class từ test_dataset (là Subset)
    # .dataset sẽ truy cập vào VOCDataset gốc
    try:
        class_map = test_dataset.dataset.class_map
        # Sắp xếp tên class theo đúng thứ tự label (1, 2, 3...)
        sorted_class_names = sorted(class_map, key=class_map.get)
        
        # Thêm 'Background' vào vị trí 0
        class_names_for_plot = ['Background'] + sorted_class_names
        # Lấy danh sách labels (0, 1, 2, 3...)
        labels_for_plot = [0] + [class_map[k] for k in sorted_class_names]

        # 3. Tạo và vẽ CM
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels_for_plot)
        plot_confusion_matrix(cm, class_names_for_plot)
    
    except Exception as e:
        print(f"[LỖI] Không thể vẽ ma trận nhầm lẫn: {e}")
        print("Kiểm tra dữ liệu CM (10 mẫu đầu):")
        print(f"y_true ({len(y_true_all)}): {y_true_all[:10]}...")
        print(f"y_pred ({len(y_pred_all)}): {y_pred_all[:10]}...")
    # ======================================================

    # Vẽ biểu đồ Loss & Accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(acc_history, marker='o', color='green')
    plt.title("Test Accuracy (IOU ≥ 0.7)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() 

    torch.save(model.state_dict(), "fasterrcnn_phone_defect.pth")
    print("✅ Training hoàn tất, model đã lưu thành 'fasterrcnn_phone_defect.pth'")
    return model, device


# =======================================================
# 3  Pipeline chính 
# =======================================================
if __name__ == "__main__":
    
    dataset_dir = "/kaggle/input/dataaaa" 

    # 1. Chỉ định thư mục chứa file .xml
    ann_dir = os.path.join(dataset_dir, "annotations")
    
    # 2. Chỉ định *danh sách* các thư mục chứa ảnh
    img_dirs = [
        os.path.join(dataset_dir, "defective"),
        os.path.join(dataset_dir, "non-phone"),
        os.path.join(dataset_dir, "non_defective")
    ]

    print(f"[INFO] Thư mục annotations: {ann_dir}")
    print(f"[INFO] Thư mục images: {img_dirs}")
    
    # 3. Tạo toàn bộ dataset
    full_dataset = VOCDataset(
        img_dirs=img_dirs, 
        ann_dir=ann_dir
    )
   
    print(f"[INFO] Tổng cộng có {len(full_dataset)} ảnh (đã khớp với XML) trong dataset.")

    # 4. Chia Train / Test 85% - 15%
    total_size = len(full_dataset)
    if total_size == 0:
        print("\nLỖI: Không tìm thấy ảnh nào khớp với file XML.")
        print("Hãy kiểm tra lại các đường dẫn:")
        print(f"  - ANN_DIR: {ann_dir}")
        print(f"  - IMG_DIRS: {img_dirs}")
        print("Và đảm bảo tên file (không có đuôi) của ảnh và XML khớp nhau.")
    else:
        test_size = math.floor(total_size * 0.15) 
        train_size = total_size - test_size       
        
        print(f"[INFO] Chia dataset: {train_size} ảnh (Train) và {test_size} ảnh (Test)")
        
        # Đặt seed để việc chia train/test cố định
        torch.manual_seed(42)
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # 5. Tạo DataLoader cho tập train
        train_loader = DataLoader(
            train_dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=lambda b: tuple(zip(*b))
        )
        
        # Lưu ý: test_dataset sẽ được lặp qua từng ảnh một trong hàm evaluate_model

        # 6. Bắt đầu huấn luyện
        model, device = train_faster_rcnn(train_loader, test_dataset, epochs=10)