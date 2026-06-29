import gdown
import os

def download_dataset_from_drive():
    """
    Hàm tự động tải bộ dữ liệu CSV từ Google Drive về máy nội bộ
    bằng File ID.
    """
    # 1. Khai báo thư mục lưu trữ (Theo đúng cấu trúc dự án của nhóm)
    save_dir = "data/raw/"
    
    # Tạo thư mục nếu nó chưa tồn tại (tránh lỗi)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Thay thế bằng ID file THẬT của bạn trên Drive
    # Ví dụ ID của file summary_data.csv
    summary_file_id = '1jEfhxgLExHVXbVs-SzFQ2pch6qbIXM7Y'
    
    # Ví dụ ID của file translation_data.csv
    translation_file_id = '1CIsjqgsE5Qnu4Cz4FxIp_Eib3myjzxve'

    # Tạo URL tải trực tiếp từ gdown
    summary_url = f'https://drive.google.com/uc?id={summary_file_id}'
    translation_url = f'https://drive.google.com/uc?id={translation_file_id}'

    # Đường dẫn file đầu ra
    summary_output = os.path.join(save_dir, "summary_data_raw.csv")
    translation_output = os.path.join(save_dir, "translation_data_raw.csv")

    # 3. Tiến hành tải
    print(" Đang tải bộ dữ liệu Tóm tắt từ Google Drive...")
    gdown.download(summary_url, summary_output, quiet=False)
    
    print(" Đang tải bộ dữ liệu Dịch thuật từ Google Drive...")
    gdown.download(translation_url, translation_output, quiet=False)

    print("\n Hoàn tất! Dữ liệu đã sẵn sàng trong thư mục data/processed/")

# Chạy thử hàm
if __name__ == "__main__":
    download_dataset_from_drive()