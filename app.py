import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

# =========================================================
# 1. ĐỌC DỮ LIỆU TÀI CHÍNH
# =========================================================
file_name = "Financial Statement Anomaly Dataset.csv"

try:
    df = pd.read_csv(file_name)
    print("\n=== [1] ĐỌC DỮ LIỆU THÀNH CÔNG ===")
except FileNotFoundError:
    print(f"\nLỗi: Không tìm thấy file '{file_name}' trong thư mục.")
    exit()

# =========================================================
# 2. CHUẨN BỊ DỮ LIỆU (FEATURES & TARGET)
# =========================================================
# X là các chỉ số tài chính dùng để dự báo (bỏ cột nhãn đi)
X = df.drop(columns=['Financial_Status'])
# y là kết quả nhãn (Normal hoặc High Risk)
y = df['Financial_Status']

# Chia dữ liệu: 80% để học (Train), 20% để kiểm tra (Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Quy mô tập dữ liệu huấn luyện ---")
print(f"Số mẫu huấn luyện (Train): {X_train.shape[0]}")
print(f"Số mẫu kiểm thử (Test): {X_test.shape[0]}\n")

# =========================================================
# 3. HUẤN LUYỆN MÔ HÌNH CÂY QUYẾT ĐỊNH (DECISION TREE)
# =========================================================
print("=== [2] ĐANG HUẤN LUYỆN MÔ HÌNH AI... ===")
# Giới hạn độ sâu bằng 3 để cây ngắn gọn, dễ giải trình cho giảng viên
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("=> Huấn luyện mô hình Decision Tree thành công!\n")

# =========================================================
# 4. ĐÁNH GIÁ ĐỘ CHÍNH XÁC CỦA MÔ HÌNH
# =========================================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=== [3] KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
print(f"Độ chính xác tổng thể (Accuracy): {accuracy * 100:.2f}%\n")
print("--- Báo cáo chi tiết hiệu suất (Classification Report) ---")
print(classification_report(y_test, y_pred))

# =========================================================
# 5. XUẤT ẢNH CÂY QUYẾT ĐỊNH ĐỂ CHO VÀO BÁO CÁO WORD
# =========================================================
plt.figure(figsize=(15, 8))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True)
plt.title("So do cay quyet dinh du doan gian lan tai chinh (Decision Tree)")

# Ép hệ thống lưu ảnh trực tiếp
plt.savefig('./so_do_cay_quyet_dinh.png', bbox_inches='tight')
print("=> ĐÃ XUẤT SƠ ĐỒ CÂY THÀNH CÔNG: 'so_do_cay_quyet_dinh.png'!")

