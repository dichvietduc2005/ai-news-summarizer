import pandas as pd
import re
import os
import html
from langdetect import detect, DetectorFactory

# Đảm bảo kết quả nhận diện ngôn ngữ luôn ổn định qua các lần chạy
DetectorFactory.seed = 0

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
    
    # Làm sạch văn bản qua 파ipeline
    df['text'] = df['text'].apply(clean_text_base)
    df['summary'] = df['summary'].apply(clean_text_base)
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
    
    # Làm sạch văn bản qua 파ipeline
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
    # Setup đường dẫn tương đối (từ src/data_prep ra ngoài thư mục gốc)
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