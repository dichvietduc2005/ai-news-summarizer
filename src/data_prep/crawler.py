import requests
from bs4 import BeautifulSoup
import re
import html
from langdetect import detect, DetectorFactory
# Bỏ comment dòng dưới nếu bạn chạy trong project gốc của bạn
# from src.data_prep.cleaner import remove_unmatched_brackets
from underthesea import word_tokenize
import nltk

# Cố định seed để langdetect luôn ổn định
DetectorFactory.seed = 0

def clean_text(raw_text: str) -> str:
    """
    Hàm làm sạch văn bản Tích Hợp (All-in-One).
    Tự động xử lý cả mã HTML (xóa thẻ rác, caption) và văn bản thuần (xóa link, regex báo chí).
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""
        
    # 1. Bắt lỗi hiển thị công thức từ Excel
    if raw_text.strip() in ["#NAME?", "#VALUE!", "#REF!"]:
        return ""

    soup = BeautifulSoup(raw_text, 'html.parser')
    
    # Băm nát các thẻ thường chứa rác (caption, video, quảng cáo, script, style)
    for unwanted in soup.find_all(['figcaption', 'video', 'audio', 'script', 'style']):
        unwanted.decompose()
        
    # Băm nát các khu vực có class mang tính chất metadata/rác
    for element in soup.find_all(class_=re.compile(r'caption|photo|avatar|pic|source|author|relate', re.IGNORECASE)):
        element.decompose()
        
    # Trích xuất text thuần (chia các block bằng \n để Regex dễ nhận diện dòng)
    text = soup.get_text(separator='\n')


    
    # Giải mã HTML entities (VD: &quot;, &amp;)
    text = html.unescape(text)
    
    # Xóa Links (URL)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.\S+', '', text)
    
    # Xóa Số điện thoại (Định dạng VN: 10 số bắt đầu bằng 0 hoặc +84)
    text = re.sub(r'\b(?:0|\+84)[3|5|7|8|9]\d{8}\b', '', text)
    
    # TẬP TỪ KHÓA ĐIỀU HƯỚNG / METADATA
    news_keywords = r"Ảnh|Video|Nguồn|Tác giả|Đồ họa|Theo|Quảng cáo|Xem thêm|Đọc thêm|Mời bạn đọc|Theo dõi|Tin liên quan|Bạn đọc quan tâm"
    
    # Xóa các đoạn có từ khóa nằm trong ngoặc
    text = re.sub(rf'[\(\[]?({news_keywords})\s*:.*?[\)\]]', '', text, flags=re.IGNORECASE)
    
    # Xóa toàn bộ dòng nếu chứa cụm từ khóa có kèm dấu hai chấm hoặc gạch nối
    text = re.sub(rf'(?im)^.*({news_keywords})\s*[:\-].*$', '', text)
    
    # Xóa toàn bộ dòng nếu bắt đầu bằng các cụm từ khóa
    text = re.sub(rf'(?im)^({news_keywords}).*$', '', text)
    
    # Xóa triệt để các cụm dạng "- Ảnh 1", "(Ảnh 2)", "Ảnh 3." lơ lửng cuối câu
    text = re.sub(r'(?i)[-\(\[]?\s*Ảnh\s*\d+[\.\)\]]?\s*', ' ', text)
    


    # Xóa các dấu ngoặc () [] {} bị mồ côi (Bỏ comment nếu dùng module của bạn)
    # text = remove_unmatched_brackets(text)
    
    # Xóa ký tự Unicode ẩn (Zero-width space)
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    
    # Gom nhiều dấu ngoặc kép thành 1 dấu
    text = re.sub(r'"+', '"', text)
    
    # Rút gọn mọi loại khoảng trắng, tab, xuống dòng thành 1 dấu cách (Chạy CUỐI CÙNG)
    text = re.sub(r'[\r\n\t\s]+', ' ', text)
    text = text.strip()
    
    # Xóa các ký tự toán học rác ở ĐẦU TIÊN của chuỗi
    text = re.sub(r'^[-=+@]+', '', text).strip()
    
    # Sửa lỗi dấu câu lơ lửng và kéo dấu câu sát vào chữ
    text = text.replace(' :', ':').replace(' ,', ',').replace(' .', '.')
    text = re.sub(r'\s+([.,!?])', r'\1', text) 
    
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
   Màn đọ sức giữa đội tuyển Ma Rốc và Hà Lan tại vòng 32 đội World Cup 2026 (sáng 30.6) đã diễn ra với kịch bản phản ánh đúng vị thế hiện tại của hai nền bóng đá. Kể từ chiến tích lịch sử lọt vào bán kết World Cup 2022 tại Qatar, Ma Rốc đã chính thức cởi bỏ cái mác "ngựa ô" để bước vào giải đấu tại Bắc Mỹ hè này với tâm thế của một kẻ thách thức khó chịu, sẵn sàng lung lay trật tự cũ của bóng đá thế giới. Trên sân cỏ, đoàn quân của HLV Mohamed Ouahbi đã chơi trên chân trước một "ông lớn" của bóng đá châu Âu. Dù Hà Lan là đội mở tỷ số trước nhờ công của Cody Gakpo, nhưng khoảng thời gian trước và sau bàn thắng đã chứng kiến sự vượt trội của đại diện Bắc Phi.

Ma Rốc đã làm tốt hơn Hà Lan ở mọi thông số. Theo thống kê của FIFA, họ áp đặt thế trận với tỷ lệ kiểm soát bóng lên tới 61%, tung ra 11 cú dứt điểm với 6 lần trúng đích, đồng thời thực hiện tới 780 đường chuyền đạt độ chính xác lên đến 92%. Những con số áp đảo này hoàn toàn bóp nghẹt một Hà Lan với 30% thời lượng giữ bóng và 356 đường chuyền, 7 pha dứt điểm (3 lần trúng đích). Trong 2 hiệp phụ, Ma Rốc là đội thể hiện nguồn năng lượng dồi dào và liên tục tổ chức vây ráp tầm cao, trong khi Hà Lan tỏ ra hụt hơi rõ rệt và có vẻ muốn kéo trận đấu vào loạt luân lưu. Do đó, việc Ma Rốc đánh bại Hà Lan không còn là một bất ngờ, mà là phần thưởng hoàn toàn xứng đáng cho đội bóng chơi hay hơn.

World Cup 2006: Đội tuyển Ma Rốc ‘biến hình’ ngoạn mục, lời khẳng định của một thế lực mới- Ảnh 1.

Saibari là hiện thân cho sự vươn mình mạnh mẽ của bóng đá Ma Rốc. Anh đã sút tung lưới Brazil, trong trận hòa 1-1 ở vòng bảng World Cup 2026

ẢNH: REUTERS
    """

    print("\n[1] VĂN BẢN GỐC (Dùng repr() để thấy rõ các ký tự ẩn và khoảng trắng):")
    print(repr(sample_raw_text)) 

    # Gọi hàm làm sạch
    cleaned_result = clean_text(sample_raw_text)

    print("\n[2] KẾT QUẢ SAU KHI LÀM SẠCH:")
    print(repr(cleaned_result))
    print("=" * 60)