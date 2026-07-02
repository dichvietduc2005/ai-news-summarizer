import pandas as pd

# Đọc file dữ liệu của nhóm
df = pd.read_csv("data/processed/summary_data.csv")

count_errors = 0

print("Đang quét dữ liệu...\n")

for index, row in df.iterrows():
    text = str(row['text']).strip()
    summary = str(row['summary']).strip()
    
    # Lấy câu đầu tiên của bài báo (tách bằng dấu chấm)
    first_sentence = text.split('.')[0].strip()
    
    # Nếu bản tóm tắt chứa y nguyên câu đầu tiên này
    if first_sentence in summary and len(first_sentence) > 10:
        count_errors += 1
        # In thử 3 dòng bị lỗi ra màn hình để bạn xem tận mắt
        if count_errors <= 3:
            print(f"--- Dòng lỗi thứ {index + 1} ---")
            print(f"Tóm tắt: {summary}")
            print("-" * 30)

print(f"\n=> KẾT QUẢ: Phát hiện {count_errors} dòng có dấu hiệu chỉ copy câu đầu!")