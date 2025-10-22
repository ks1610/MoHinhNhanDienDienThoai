import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
# <<< THÊM MỚI: Import confusion_matrix và ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# >>> KẾT THÚC THÊM MỚI
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import torch
from skimage.feature import hog
# from skimage import exposure # Bỏ comment này nếu bạn muốn xem ảnh HOG

# =========================
# 1. HÀM TRÍCH XUẤT ĐẶC TRƯNG HOG (THAY THẾ)
# =========================
def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Không thể đọc ảnh: {img_path}")
        return None

 
    img = cv2.resize(img, (128, 128))

   
    features, hog_image = hog(img, 
                              orientations=9, 
                              pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), 
                              visualize=True, 
                              block_norm='L2-Hys')

   

   
    std_dev = np.std(img)
    
    
    combined_features = np.hstack((features, std_dev))
    
    return combined_features


# =========================
# 2. HÀM TẢI DỮ LIỆU TỪ CÁC THƯ MỤC ĐÃ CHIA (ĐÃ SỬA LỖI ĐỌC THƯ MỤC CON)
# =========================
def load_data_from_split(dataset_path, split_name, label_map):
    X, y, image_paths = [], [], []
    split_dir = os.path.join(dataset_path, split_name) # vd: /kaggle/input/dataset/train

    if not os.path.exists(split_dir):
        print(f"⚠️ Không tìm thấy thư mục: {split_dir}")
        return np.array(X), np.array(y), image_paths

    # Lặp qua các thư mục lớp chính ('non-phone', 'defective', ...)
    for label_name, label_id in label_map.items():
        class_dir = os.path.join(split_dir, label_name)
        if not os.path.exists(class_dir):
            print(f"⚠️ Không tìm thấy thư mục con: {class_dir}")
            continue

        # Lặp thêm một cấp thư mục con bên trong (vd: 'hop', 'tivi', 'screen-on')
        for sub_dir_name in os.listdir(class_dir):
            sub_dir_path = os.path.join(class_dir, sub_dir_name)
            
            # Đảm bảo rằng đó là một thư mục
            if os.path.isdir(sub_dir_path):
                # Bây giờ mới lặp qua các file ảnh bên trong thư mục con này
                for filename in os.listdir(sub_dir_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(sub_dir_path, filename)
                        features = extract_features(file_path)
                        
                        if features is not None:
                            X.append(features)
                            y.append(label_id)
                            image_paths.append(file_path)
            else:
                # Nếu có file ảnh nằm ngay ngoài, cũng đọc luôn
                if sub_dir_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(class_dir, sub_dir_name)
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(label_id)
                        image_paths.append(file_path)

    return np.array(X), np.array(y), image_paths



# =========================
# 3. VẼ PHÂN BỐ DỮ LIỆU 
# =========================
def plot_dataset_distribution(train_size, val_size, test_size):
    labels = ['Train', 'Validation', 'Test']
    sizes = [train_size, val_size, test_size]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, sizes, color=['#4CAF50', '#FFC107', '#2196F3'])
    plt.title('Phân bố dữ liệu Train / Validation / Test')
    plt.ylabel('Số lượng mẫu')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), ha='center', va='bottom')
    plt.show()


# =========================
# 4. DỰ ĐOÁN ẢNH MỚI 
# =========================
def predict_image(model, img_path, class_names):
    features = extract_features(img_path)
    if features is None:
        return
    
    if features.ndim == 1:
        features = features.reshape(1, -1)
        
    result_id = model.predict(features)[0]
    result_name = class_names[result_id]
    
    if result_id == 2: # 'defective'
        print(f"📸 {os.path.basename(img_path)} → ⚠️ BỊ VỠ ({result_name})")
    elif result_id == 1: # 'non defective'
        print(f"📸 {os.path.basename(img_path)} → ✅ BÌNH THƯỜNG ({result_name})")
    else: # 'non-phone'
        print(f"📸 {os.path.basename(img_path)} → ❌ KHÔNG PHẢI ĐIỆN THOẠI ({result_name})")


# =========================
# 5. HÀM VẼ BIỂU ĐỒ BẰNG PCA
# =========================
def plot_svm_pca_boundary(X_train, y_train, X_test, y_test, class_names):
    print("\n🎨 Đang giảm chiều dữ liệu bằng PCA để vẽ biểu đồ...")
    
  
    X_full = np.vstack((X_train, X_test))
    y_full = np.hstack((y_train, y_test))
    
 
    pca = PCA(n_components=2)
    X_pca_full = pca.fit_transform(X_full)
    

    svm_2d = SVC(kernel='rbf', C=10).fit(X_pca_full, y_full)
    
    x_min, x_max = X_pca_full[:, 0].min() - 1, X_pca_full[:, 0].max() + 1
    y_min, y_max = X_pca_full[:, 1].min() - 1, X_pca_full[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    
    scatter = plt.scatter(X_pca_full[:, 0], X_pca_full[:, 1], c=y_full, cmap='viridis', edgecolors='k', alpha=0.7)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes")
    
    plt.title('Biểu đồ PCA 2-chiều của đặc trưng HOG và ranh giới SVM (minh họa)')
    plt.xlabel('Thành phần chính 1 (PC1)')
    plt.ylabel('Thành phần chính 2 (PC2)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# =========================
# 6.  HÀM VẼ MA TRẬN NHẦM LẪN
# =========================
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """
    Hàm này vẽ ma trận nhầm lẫn.
    """
  
    cm = confusion_matrix(y_true, y_pred)
    
   
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
   
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(title)
    plt.grid(False) # Tắt lưới của matplotlib để hiển thị rõ hơn
    plt.show()

# =========================
# 7. CHẠY TOÀN BỘ
# =========================
model_file = "/kaggle/working/svm_model.pth"
dataset_base_path = '/kaggle/input/dataset'

# Định nghĩa các lớp (khớp với tên thư mục của bạn)
label_map = {
    'non-phone': 0,
    'non-defective': 1,
    'defective': 2
}
class_names = list(label_map.keys())

print("🔸 Đang tải dữ liệu train...")
X_train, y_train, paths_train = load_data_from_split(dataset_base_path, 'train', label_map)
print(f"✅ Đã tải {len(X_train)} ảnh train.")

print("🔸 Đang tải dữ liệu val...")
X_val, y_val, paths_val = load_data_from_split(dataset_base_path, 'val', label_map)
print(f"✅ Đã tải {len(X_val)} ảnh val.")

print("🔸 Đang tải dữ liệu test...")
X_test, y_test, paths_test = load_data_from_split(dataset_base_path, 'test', label_map)
print(f"✅ Đã tải {len(X_test)} ảnh test.")

if len(X_train) == 0:
    raise ValueError("❌ Không có dữ liệu train để huấn luyện. Vui lòng kiểm tra lại đường dẫn.")

print(f"\n📊 Kích thước tập:")
print(f"  - Train: {len(X_train)}")
print(f"  - Validation: {len(X_val)}")
print(f"  - Test: {len(X_test)}")

plot_dataset_distribution(len(X_train), len(X_val), len(X_test))

# =========================
# 8. HUẤN LUYỆN BẰNG GRIDSEARCHCV
# =========================
model = None
if os.path.exists(model_file):
    print(f"\n💾 Đang tải mô hình đã huấn luyện từ {model_file} ...")
    model = torch.load(model_file)
else:
    print("\n🚀 Bắt đầu huấn luyện mô hình và tìm tham số tốt nhất (GridSearchCV)...")
    
    # Định nghĩa không gian tham số để tìm
    # Chúng ta tìm 'C' (độ phạt) và 'gamma' (ảnh hưởng của 1 điểm)
    param_grid = {
        'C': [1, 10, 100],            # Thử các giá trị C
        'gamma': ['scale', 0.1, 0.01], # Thử các giá trị gamma
        'kernel': ['rbf']
    }
    
    # cv=3: Chia tập train thành 3 phần để kiểm tra chéo, tăng độ tin cậy
    # n_jobs=-1: Dùng tất cả CPU để chạy song song, nhanh hơn
    grid_search = GridSearchCV(SVC(decision_function_shape='ovr'), 
                               param_grid, 
                               cv=3, 
                               verbose=2, 
                               n_jobs=-1)
    
    # Huấn luyện trên TẬP TRAIN + VALIDATION (gộp lại để có nhiều dữ liệu hơn)
    X_train_full = np.vstack((X_train, X_val))
    y_train_full = np.hstack((y_train, y_val))
    
    print(f"Huấn luyện trên {len(X_train_full)} mẫu (Train + Val)...")
    grid_search.fit(X_train_full, y_train_full)
    
    print("\n🎉 ĐÃ HUẤN LUYỆN XONG!")
    print(f"Tham số tốt nhất tìm được: {grid_search.best_params_}")
    
    # Lấy ra mô hình tốt nhất
    model = grid_search.best_estimator_
    
    # Đánh giá lại trên tập Validation (chỉ để tham khảo)
    y_val_pred = model.predict(X_val)
    print("\n=== 📌 KẾT QUẢ VALIDATION (với mô hình tốt nhất) ===")
    print(classification_report(y_val, y_val_pred, target_names=class_names))

    # <<< THÊM MỚI: Vẽ ma trận nhầm lẫn cho tập Validation
    plot_confusion_matrix(y_val, y_val_pred, class_names, "Ma trận nhầm lẫn - TẬP VALIDATION")
    # >>> KẾT THÚC THÊM MỚI

    # Lưu mô hình tốt nhất
    torch.save(model, model_file)
    print(f"💾 Đã lưu mô hình tốt nhất vào: {model_file} (dạng .pth)")

# =========================
# 9. ĐÁNH GIÁ TRÊN TEST SET
# =========================
if model:
    print("\n=== 🧪 KẾT QUẢ TRÊN TEST SET ===")
    y_test_pred = model.predict(X_test)
    print("Độ chính xác:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # <<< THÊM MỚI: Vẽ ma trận nhầm lẫn cho tập Test
    plot_confusion_matrix(y_test, y_test_pred, class_names, "Ma trận nhầm lẫn - TẬP TEST")
    # >>> KẾT THÚC THÊM MỚI

    # Vẽ biểu đồ PCA
    plot_svm_pca_boundary(X_train, y_train, X_test, y_test, class_names)
else:
    print("❌ Không có mô hình để đánh giá.")