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

def crawl_article(url: str) -> str:
    """
    Hàm cào dữ liệu báo chí từ URL bằng BeautifulSoup.
    Tự động trích xuất nội dung chính từ các thẻ paragraph (<p>).
    """
    if not url or not url.startswith(("http://", "https://")):
        return ""
        
    try:
        # Cấu hình Header giả lập trình duyệt để tránh bị các báo block IP
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding # Tự động sửa lỗi font tiếng Việt
        
        if response.status_code != 200:
            return ""
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Tìm tất cả các thẻ <p> (thẻ đoạn văn phổ biến của bài báo)
        paragraphs = soup.find_all('p')
        
        text_content = []
        for p in paragraphs:
            p_text = p.get_text().strip()
            # Lọc bỏ các đoạn quá ngắn (thường là text của nút bấm, menu hoặc quảng cáo ẩn)
            if len(p_text.split()) > 8:
                text_content.append(p_text)
                
        # Gộp các đoạn văn lại với nhau ngăn cách bằng dấu xuống dòng
        return "\n".join(text_content)
        
    except Exception as e:
        print(f"[Crawler Error] Lỗi khi cào link {url}: {str(e)}")
        return ""

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