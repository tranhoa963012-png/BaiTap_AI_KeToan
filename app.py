from flask import Flask, render_template, request
import pandas as pd
import os
from dotenv import load_dotenv

# Tải cấu hình bảo mật từ file .env theo đúng yêu cầu bài tập
load_dotenv()
secret_key = os.getenv("SECRET_KEY")

app = Flask(__name__)

# Tên chính xác tệp dữ liệu lớn trong thư mục của bạn
DATA_FILE = "Financial Statement Anomaly Dataset.csv"

def analyze_financial_data(assets, liabilities, revenue):
    # Kiểm tra xem file dữ liệu lớn có nằm đúng vị trí không
    if not os.path.exists(DATA_FILE):
        return "LỖI: Không tìm thấy tệp dữ liệu lớn 'Financial Statement Anomaly Dataset.csv'!", "#ff4757"
    
    try:
        # Sử dụng thư viện Pandas để đọc tệp dữ liệu lớn CSV
        df = pd.read_csv(DATA_FILE)
        
        # 1. Tính toán tỷ số Nợ / Tài sản của doanh nghiệp bạn vừa nhập
        current_debt_ratio = liabilities / assets if assets > 0 else 0
        
        # 2. Tự động nhận diện tên cột trong file dữ liệu của bạn
        col_assets = 'Total_Assets' if 'Total_Assets' in df.columns else df.columns[0]
        col_liab = 'Total_Liabilities' if 'Total_Liabilities' in df.columns else df.columns[1]
        
        # 3. Quét qua toàn bộ tệp dữ liệu lớn để tính chỉ số trung bình ngành
        df['Debt_Ratio'] = df[col_liab] / df[col_assets]
        avg_industry_ratio = df['Debt_Ratio'].mean()
            
        # 4. Thuật toán đối chiếu dữ liệu thực tế:
        if liabilities > assets:
            return "❌ CẢNH BÁO NGUY HIỂM: Tổng nợ đang lớn hơn Tổng tài sản! Doanh nghiệp mất cân đối tài chính nghiêm trọng.", "#ff4757"
        elif current_debt_ratio > avg_industry_ratio * 1.5:
            return f"⚠️ CẢNH BÁO BẤT THƯỜNG: Tỷ số Nợ/Tài sản của bạn ({current_debt_ratio:.2f}) cao vượt ngưỡng 50% so với mức trung bình ngành thực tế ({avg_industry_ratio:.2f}) trích xuất từ tệp dữ liệu lớn!", "#ffa502"
        else:
            return f"✅ KẾT QUẢ AN TOÀN: Tỷ số Nợ/Tài sản ({current_debt_ratio:.2f}) nằm trong phạm vi ổn định so với trung bình dữ liệu ngành ({avg_industry_ratio:.2f}).", "#2ed573"
            
    except Exception as e:
        return f"Lỗi hệ thống khi xử lý dữ liệu lớn: {str(e)}", "#ff4757"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy đầy đủ dữ liệu từ 3 ô trên giao diện web bạn nhập vào
        assets = float(request.form['Total_Assets'])
        liabilities = float(request.form['Total_Liabilities'])
        revenue = float(request.form['Revenue'])
        
        # Chạy hàm đối chiếu với tệp dữ liệu lớn CSV
        ket_qua, mau_sac = analyze_financial_data(assets, liabilities, revenue)
        
        return f"""
        <body style="background: #1e3c72; color: white; font-family: sans-serif; text-align: center; padding-top: 100px;">
            <div style="background: rgba(255,255,255,0.1); display: inline-block; padding: 40px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2); max-width: 600px;">
                <h2 style="color: {mau_sac}; line-height: 1.6;">{ket_qua}</h2>
                <p style="color: #ccc; margin-top: 20px;">Số liệu phân tích dựa trên đầu vào: Tài sản={assets:,} | Nợ={liabilities:,} | Doanh thu={revenue:,}</p>
                <br><br>
                <a href="/" style="background: #00f2fe; color: #1e3c72; padding: 10px 20px; text-decoration: none; font-weight: bold; border-radius: 5px;">Quay lại kiểm tra tiếp</a>
            </div>
        </body>
        """
    except:
        return "Vui lòng nhập đúng số liệu định dạng số!"

if __name__ == '__main__':
    app.run(debug=True)
    