# Đồ án Trí tuệ Nhân tạo: Hệ thống Tóm tắt và Dịch thuật

Dự án này sử dụng TextRank, ViT5, MarianMT và NER để tóm tắt, dịch thuật và trích xuất thực thể từ bài báo.

## Cài đặt
```bash
pip install -r requirements.txt
```

## Chạy dự án
1. Chạy Backend (FastAPI):
```bash
uvicorn api.main:app --reload
```

2. Chạy Frontend (Streamlit):
```bash
streamlit run app/frontend.py
```

## Phân công công việc chi tiết
- **Anh Quốc (Data Engineer & NLP Preprocessing)**: Lo "cửa ngõ" vào của hệ thống. Cào dữ liệu, làm sạch, tách từ (Underthesea/NLTK). Nhận diện ngôn ngữ (Anh/Việt). **Đặc biệt:** Chuẩn bị 2 bộ Dataset (1 cho tóm tắt, 1 song ngữ cho dịch thuật) và viết hàm tính ROUGE/BLEU. (`src/data_prep/`)
- **Viết Đức (Information Extraction & Graphs)**: Phụ trách Thuật toán và Đồ thị. Viết thuật toán TextRank để cắt giảm độ dài văn bản. Tích hợp mô hình NER để trích xuất Thực thể. Dùng NetworkX/Pyvis vẽ Đồ thị tri thức (Knowledge Graph). (`src/algorithms/`)
- **Trọng Nghĩa (LLM Summarization Specialist)**: Xử lý trái tim hệ thống. Tinh chỉnh (Fine-tune) mô hình tóm tắt ViT5 và mT5-XLSum. Nhận dữ liệu từ Đức để đưa vào model tránh tràn 512 tokens. Tối ưu hóa RAM/VRAM. (`src/models/summarizer.py`)
- **Trung Kiên (Nhóm trưởng - MLOps & System Architect)**: Mảnh ghép cuối cùng. **Huấn luyện tinh chỉnh (Fine-tune)** mô hình dịch MarianMT. Viết API (FastAPI) nối code của Quốc, Đức, Nghĩa lại. Làm giao diện Streamlit và đóng gói Docker. (`src/models/translator.py`, `api/`, `app/`)
