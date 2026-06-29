import requests
from bs4 import BeautifulSoup
import re
import html
from langdetect import detect, DetectorFactory
from src.data_prep.cleaner import remove_unmatched_brackets
from underthesea import word_tokenize # Dành cho tiếng Việt
import nltk # Dành cho tiếng Anh

# Cố định seed để langdetect luôn ổn định
DetectorFactory.seed = 0

def clean_text(raw_text: str) -> str:
    """
    Hàm làm sạch văn bản (xóa HTML, ký tự đặc biệt, sửa lỗi ngoặc mồ côi và Excel Injection).
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""
        
    # 1. Bắt lỗi hiển thị công thức từ Excel
    if raw_text.strip() in ["#NAME?", "#VALUE!", "#REF!"]:
        return ""
        
    # 2. Giải mã HTML entities (VD: &quot;, &amp;)
    text = html.unescape(raw_text)
    
    # 3. Xóa các dấu ngoặc () [] {} bị mồ côi bằng Stack
    text = remove_unmatched_brackets(text)
    
    # 4. Xóa ký tự Unicode ẩn (Zero-width space)
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    
    # 5. Gom nhiều dấu ngoặc kép thành 1 dấu (vd: """ -> ")
    text = re.sub(r'"+', '"', text)
    
    # 6. Rút gọn mọi loại khoảng trắng, tab, xuống dòng thành 1 dấu cách
    text = re.sub(r'[\r\n\t\s]+', ' ', text)
    text = text.strip()
    
    # 7. Xóa các ký tự toán học rác ở ngay ĐẦU TIÊN của chuỗi để chặn lỗi hiển thị CSV
    text = re.sub(r'^[-=+@]+', '', text).strip()
    
    # 8. Sửa lỗi dấu câu lơ lửng
    text = text.replace(' :', ':').replace(' ,', ',').replace(' .', '.')
    
    return text.strip(' "')

def detect_language(text: str) -> str:
    """
    Hàm phát hiện ngôn ngữ tự động sử dụng langdetect.
    - Output: 'vi' hoặc 'en'
    """
    if not text or not text.strip():
        return "vi" # Mặc định trả về vi làm phương án an toàn
        
    try:
        lang = detect(text)
        if lang in ["vi", "en"]:
            return lang
        return "vi" # Nếu ra ngôn ngữ khác, fallback về 'vi' cho dự án
    except:
        return "vi"
def tokenize_words(text: str, language: str = 'vi') -> list:
    """
    Hàm tách từ chuyên sâu (Word Tokenization) phục vụ cho thống kê,
    làm TextRank truyền thống hoặc trích xuất Đồ thị tri thức (NER).
    """
    if not text or not text.strip():
        return []
        
    if language == 'en':
        # Tách từ tiếng Anh bằng NLTK
        return nltk.word_tokenize(text)
    else:
        # Tách từ tiếng Việt bằng Underthesea (giữ nguyên khoảng trắng của từ phức)
        # Kết quả: ["Trí tuệ", "nhân tạo", "đang", "phát triển"]
        return word_tokenize(text)

def tokenize_words_for_graph(text: str, language: str = 'vi') -> str:
    """
    Hàm tách từ nối dấu gạch dưới phục vụ riêng cho Đồ thị Tri thức / NER.
    Kết quả: "Trí_tuệ nhân_tạo đang phát_triển"
    """
    if language == 'en' or not text:
        return text
    return word_tokenize(text, format="text")
if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Kiểm tra hàm làm sạch văn bản clean_text")
    print("=" * 60)

    # Đoạn văn bản mẫu cố tình chứa nhiều lỗi rác phổ biến từ Crawler
    sample_raw_text = """
   tôi muốn lấy thông tin của những người có cùng khóa học đã đăng ký đó tạo 1 danh sách sửa lại chức năng chat socket 1-1 cho những người có chung khóa học bao, giao diện chat như hình bên trái là list người chung khóa hoc (gồn student , giáo viên , phụ huynh , miễn là có trong khóa học) , bên phải là giao diện chat và tạo 1 nút chat ở StudentLearningPage.jsx để mỗi khóa có thể nhấn vào và hiện lên list danh sách khóa học đó
    """

    print("\n[1] VĂN BẢN GỐC (Dùng repr() để thấy rõ các ký tự ẩn và khoảng trắng):")
    print(repr(sample_raw_text)) 

    # Gọi hàm làm sạch
    cleaned_result = clean_text(sample_raw_text)

    print("\n[2] KẾT QUẢ SAU KHI LÀM SẠCH:")
    print(repr(cleaned_result))
    print("=" * 60)