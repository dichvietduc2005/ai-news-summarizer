import pandas as pd
import re
import os
import html
from langdetect import detect, DetectorFactory

# Đảm bảo kết quả nhận diện ngôn ngữ luôn ổn định qua các lần chạy
DetectorFactory.seed = 0

def clean_news_noise(text):
    """
    Bổ sung các bộ lọc nhiễu đặc thù của báo chí (Dành riêng cho tập Summary):
    - Xóa link, email, số điện thoại toàn cục.
    - Duyệt từng dòng để xóa quảng cáo (Xem thêm, Đọc thêm...), caption ảnh, dòng tác giả, thời gian đăng.
    """
    if not isinstance(text, str):
        return text

    # 1. Xóa link (Xử lý cả http, https và www.)
    text = re.sub(r'https?://\S+|www\.\S+|http\S+', '', text)

    # 2. Xóa email
    text = re.sub(r'\S+@\S+', '', text)

    # 3. Xóa số điện thoại (Bắt đầu bằng số 0, dài từ 10-11 chữ số)
    text = re.sub(r'\b0\d{9,10}\b', '', text)

    text = re.sub(
    r'[^.!?]{10,200}\s+ẢNH:\s*[@A-Za-zÀ-Ỹà-ỹ0-9_.\- ]{2,80}',
    '. ',
    text,
    flags=re.IGNORECASE
    )

    # Xử lý theo từng dòng để tránh toán tử .* xóa lẹm sang các đoạn văn khác
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        # 4. Xóa dòng thời gian đăng (VD: 27/06/2026 10:35 GMT+7 hoặc 29-06-2026)
        if re.match(r'^\d{2}[/\-]\d{2}[/\-]\d{4}', line_str):
            continue

        # 5. Xóa caption ảnh, nguồn, đồ họa, tác giả ở đầu dòng (VD: Ảnh: Nhật Thịnh, Tác giả: Minh Hải)
        if re.match(r'^(Ảnh|Nguồn|Đồ họa|Tác giả)[:\s]', line_str, re.IGNORECASE):
            continue

        # 6. Xóa dòng trích dẫn thông tấn (VD: Theo Reuters, Theo AFP, Theo TTXVN)
        if re.match(r'^Theo\s+(Reuters|AFP|TTXVN|Thanh Niên|Tuổi Trẻ)\b', line_str, re.IGNORECASE):
            continue

        # 7. Xóa quảng cáo / noise lơ lửng (Xem thêm..., Đọc thêm..., Tin liên quan...) đến hết dòng
        line_str = re.sub(r'(Xem thêm|Đọc thêm|Mời bạn đọc|Theo dõi|Tin liên quan|Video|Bạn đọc quan tâm).*', '', line_str, flags=re.IGNORECASE)
        
        # Nếu sau khi lọc dòng đó vẫn còn chữ thì giữ lại
        if line_str.strip():
            cleaned_lines.append(line_str.strip())

    return '\n'.join(cleaned_lines)

def remove_unmatched_brackets(text):
    """
    Sử dụng Stack để tìm và xóa các dấu ngoặc (), [], {} không có cặp.
    Ngăn chặn mô hình AI học phải các ký tự rác mồ côi.
    """
    if not isinstance(text, str):
        return text
        
    stack = []
    to_remove = set()
    pairs = {')': '(', ']': '[', '}': '{'}

    # Quét từng ký tự trong chuỗi
    for i, char in enumerate(text):
        if char in pairs.values(): # Dấu mở
            stack.append((char, i))
        elif char in pairs.keys(): # Dấu đóng
            if stack and stack[-1][0] == pairs[char]:
                stack.pop() # Có cặp -> Hợp lệ
            else:
                to_remove.add(i) # Không có mở tương ứng -> Đánh dấu rác

    # Dấu mở còn sót lại trong stack cũng là rác
    for _, i in stack:
        to_remove.add(i)

    # Lắp ráp lại chuỗi bỏ qua rác
    cleaned_chars = [char for i, char in enumerate(text) if i not in to_remove]
    return ''.join(cleaned_chars)

def clean_text_base(text):
    """Bước làm sạch văn bản chung, tích hợp mọi chốt chặn"""
    if not isinstance(text, str):
        return ""
        
    # 1. Bắt lỗi nếu file đã lỡ bị lưu đè thành lỗi công thức từ Excel
    if text.strip() in ["#NAME?", "#VALUE!", "#REF!"]:
        return ""
    
    # 2. Giải mã HTML entities sót lại (VD: &quot;, &amp;)
    text = html.unescape(text)
    
    # --- BỔ SUNG TẠI ĐÂY ---
    # Xóa các tag múi giờ rác nằm trong ngoặc vuông hoặc ngoặc đơn dạng [UTC+200], (UTC+7), [GMT+7]
    text = re.sub(r'[\[\(](UTC|GMT)[+\-]?\d+[\]\)]', '', text, flags=re.IGNORECASE)
    # -----------------------
    
    # 3. Xóa các dấu ngoặc () [] {} bị mồ côi bằng Stack
    text = remove_unmatched_brackets(text)
    
    # 4. Xóa ký tự Unicode vô hình (Zero-width space)
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    
    # 5. Gom nhiều dấu ngoặc kép thành 1 dấu (vd: """ -> ")
    text = re.sub(r'"+', '"', text)
    
    # 6. Rút gọn mọi loại khoảng trắng, tab, xuống dòng thành 1 dấu cách
    text = re.sub(r'[\r\n\t\s]+', ' ', text)
    text = text.strip()
    
    # 7. SANITIZE EXCEL/CSV INJECTION: Xóa =, +, -, @ ở ngay ĐẦU TIÊN của chuỗi
    text = re.sub(r'^[-=+@]+', '', text).strip()
    
    # 8. Sửa lỗi dấu câu lơ lửng sau khi xóa ngoặc/từ (Optional)
    text = text.replace(' :', ':').replace(' ,', ',').replace(' .', '.')
    
    # Lột sạch khoảng trắng và dấu ngoặc kép ở 2 đầu chuỗi
    return text.strip(' "')

def is_valid_language(text, expected_lang):
    """Kiểm tra ngôn ngữ với bộ bắt lỗi try-except an toàn"""
    try:
        if len(text.split()) < 3: 
            return True 
        return detect(text) == expected_lang
    except:
        return False

def clean_summary_dataset(input_path, output_path):
    print(f"Bắt đầu làm sạch tập tóm tắt: {input_path}")
    df = pd.read_csv(input_path)
    
    # Drop NA & Duplicates
    df = df.dropna().drop_duplicates()
    
    # PIPELINE CẢI TIẾN: Lọc nhiễu báo chí trước -> Sau đó chuẩn hóa base
    df['text'] = df['text'].apply(lambda x: clean_text_base(clean_news_noise(x)))
    df['summary'] = df['summary'].apply(lambda x: clean_text_base(clean_news_noise(x)))
    df = df[(df['text'] != '') & (df['summary'] != '')]
    
    # Lọc logic độ dài (Bản gốc phải >= 30 từ, và dài hơn bản tóm tắt)
    df['text_word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['summary_word_count'] = df['summary'].apply(lambda x: len(x.split()))
    
    valid_logic_mask = (df['text_word_count'] >= 30) & (df['summary_word_count'] < df['text_word_count'])
    df = df[valid_logic_mask]
    
    # Lưu file
    clean_df = df[['text', 'summary']]
    clean_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"-> Đã lưu {len(clean_df)} dòng sạch vào {output_path}\n")

def clean_translation_dataset(input_path, output_path):
    print(f"Bắt đầu làm sạch tập dịch thuật: {input_path}")
    df = pd.read_csv(input_path)
    
    # Drop NA & Duplicates
    df = df.dropna().drop_duplicates()
    
    # Tập dịch thuật chỉ chạy qua bộ clean base (Giữ nguyên gốc để tránh mất cấu trúc câu song ngữ)
    df['vi_text'] = df['vi_text'].apply(clean_text_base)
    df['en_text'] = df['en_text'].apply(clean_text_base)
    df = df[(df['vi_text'] != '') & (df['en_text'] != '')]
    
    # Lọc tỷ lệ độ dài câu (0.5 đến 1.8)
    df['vi_len'] = df['vi_text'].apply(lambda x: len(x.split()))
    df['en_len'] = df['en_text'].apply(lambda x: len(x.split()))
    df['length_ratio'] = df['vi_len'] / df['en_len']
    
    ratio_mask = (df['length_ratio'] >= 0.5) & (df['length_ratio'] <= 1.8)
    df = df[ratio_mask]
    
    # Kiểm tra ngôn ngữ
    print("Đang chạy Langdetect để kiểm tra chính tả song ngữ (Có thể mất chút thời gian)...")
    df = df[df['vi_text'].apply(lambda x: is_valid_language(x, 'vi'))]
    df = df[df['en_text'].apply(lambda x: is_valid_language(x, 'en'))]
    
    # Lưu file
    clean_df = df[['vi_text', 'en_text']]
    clean_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"-> Đã lưu {len(clean_df)} dòng song ngữ sạch vào {output_path}\n")

if __name__ == "__main__":
    # Setup đường dẫn tương đối
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    
    # Tạo thư mục processed nếu chưa có
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    summary_raw = os.path.join(RAW_DIR, "summary_data_raw.csv")
    summary_clean = os.path.join(PROCESSED_DIR, "summary_data.csv")
    
    translation_raw = os.path.join(RAW_DIR, "translation_data_raw.csv")
    translation_clean = os.path.join(PROCESSED_DIR, "translation_data.csv")
    
    if os.path.exists(summary_raw):
        clean_summary_dataset(summary_raw, summary_clean)
    else:
        print(f"Không tìm thấy dữ liệu thô tại: {summary_raw}")
        
    if os.path.exists(translation_raw):
        clean_translation_dataset(translation_raw, translation_clean)
    else:
        print(f"Không tìm thấy dữ liệu thô tại: {translation_raw}")