import pandas as pd
import re
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# 1. Danh sách từ khóa chuyên ngành
DOMAIN_KEYWORDS = [
    "UBND", "HĐND", "Trung ương Đảng", "Thủ tướng", "Chính quyền",
    "Bộ GD-ĐT", "Bộ Giáo dục", "THPT Quốc gia", "ĐHQG", "Đại học Quốc gia", "Xã hội hóa giáo dục",
    "NHNN", "Ngân hàng Nhà nước", "DNNN", "Doanh nghiệp nhà nước", "FDI", "BOT",
    "CSGT", "Cảnh sát giao thông", "Bộ Công an", "Khởi tố", "Tạm giam", "Viện Kiểm sát",
    "Xử lý nghiêm", "Đẩy mạnh", "Tăng cường", "Đồng bộ"
]

def process_and_filter_dataset():
    print("1. Đang tải dữ liệu PhoMT từ Hugging Face...")
    dataset = load_dataset("ura-hcmut/PhoMT", split="train")
    df = dataset.to_pandas()
    df = df.dropna(subset=['vi', 'en'])

    print("\n2. Đang lọc sơ bộ (độ dài <= 80 chữ & chứa từ khóa chuyên ngành)...")
    # Lọc độ dài
    df = df[
        (df['vi'].str.count(' ') + 1 <= 80) & 
        (df['en'].str.count(' ') + 1 <= 80)
    ]
    
    # Lọc từ khóa
    pattern = re.compile(r'\b(' + '|'.join(DOMAIN_KEYWORDS) + r')\b', flags=re.IGNORECASE)
    df_filtered = df[df['vi'].str.contains(pattern, na=False)].copy()
    
    print(f"-> Số câu vượt qua vòng loại sơ bộ: {len(df_filtered):,} câu.")

    if len(df_filtered) == 0:
        print("Không tìm thấy câu nào. Hãy kiểm tra lại từ khóa!")
        return

    print("\n3. Bật mô hình AI LaBSE để chấm điểm ngữ nghĩa...")
    print("-> (Lưu ý: Lần đầu chạy sẽ mất vài phút để tải mô hình LaBSE ~1.8GB về máy)")
    
    # Khởi tạo LaBSE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Đang chạy LaBSE trên: {device.upper()}")
    model = SentenceTransformer('sentence-transformers/LaBSE', device=device)

    # Chuyển dữ liệu thành list để đưa vào mô hình
    vi_sentences = df_filtered['vi'].tolist()
    en_sentences = df_filtered['en'].tolist()

    print("\n4. Đang tính toán Vector và độ tương đồng (Cosine Similarity)...")
    # Encode câu thành các ma trận vector (Tensors)
    embeddings_vi = model.encode(vi_sentences, convert_to_tensor=True, show_progress_bar=True)
    embeddings_en = model.encode(en_sentences, convert_to_tensor=True, show_progress_bar=True)

    # Tính toán Cosine Similarity từng cặp câu một (1-to-1)
    cosine_scores = torch.nn.functional.cosine_similarity(embeddings_vi, embeddings_en)
    
    # Gán điểm vào DataFrame
    df_filtered['similarity_score'] = cosine_scores.cpu().numpy()

    print("\n5. Đang lọc các cặp câu đạt chuẩn xuất sắc (Điểm >= 0.85)...")
    # Chỉ lấy những câu có điểm ngữ nghĩa từ 0.85 trở lên
    df_high_quality = df_filtered[df_filtered['similarity_score'] >= 0.85]
    
    final_count = len(df_high_quality)
    print(f"-> Số câu đạt chuẩn xuất sắc: {final_count:,} câu.")

    print("\n6. Đang trích xuất 5000 dòng và lưu file...")
    # Lấy 5000 câu ngẫu nhiên hoặc lấy hết nếu không đủ 5000
    if final_count > 5000:
        df_final = df_high_quality.sample(n=5000, random_state=42)
    else:
        df_final = df_high_quality
        print(f"-> Lưu ý: Chỉ gom được {final_count} câu siêu chuẩn, chưa đủ 5000. Bạn cần thêm từ khóa để tìm thêm.")

    # Đổi tên cột đúng chuẩn và chỉ giữ lại vi_text, en_text
    df_final = df_final.rename(columns={'vi': 'vi_text', 'en': 'en_text'})
    df_final = df_final[['vi_text', 'en_text']] # Bỏ đi cột điểm similarity để file CSV sạch sẽ theo yêu cầu
    
    # Xuất file CSV
    output_file = 'ai_training_data_labse_085.csv'
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n🎉 HOÀN THÀNH TẬP DỮ LIỆU CHẤT LƯỢNG CAO!")
    print(f"-> Đã lưu {len(df_final)} câu vào file: {output_file}")

if __name__ == "__main__":
    process_and_filter_dataset()