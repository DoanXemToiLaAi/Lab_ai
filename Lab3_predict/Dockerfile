# Sử dụng Python 3.9 làm base image
FROM python:3.9

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các file cần thiết vào container
COPY app.py model.h5 requirements.txt ./
COPY templates/ templates/

# Cài đặt thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Chạy Flask API
CMD ["python", "app.py"]
