import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import concurrent.futures
from sentence_transformers import SentenceTransformer, util

# Cấu hình cơ bản
SITEMAP_INDEX = "https://vietnamnet.vn/sitemap.xml"
TARGET_ROWS = 10000
OUTPUT_FILE = "dataset.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

# Khởi tạo Model đo Semantic Similarity (Load 1 lần duy nhất)
print("⏳ Đang tải model Semantic Similarity (vui lòng đợi vài giây)...")
SIMILARITY_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("✅ Tải model thành công!")

# -----------------------
# CLEAN TEXT
# -----------------------
def clean(text):
    if not text:
        return ""
    
    # 1. Xóa các thẻ HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Xóa các dấu >> và <<
    text = re.sub(r'>>|<<', ' ', text)
    
    # Mẹo thêm: Trên các báo đôi khi họ dùng ký tự ngoặc kép dạng « và » 
    # Nếu muốn xóa luôn cả « và », bạn thay dòng trên bằng dòng dưới đây:
    # text = re.sub(r'>>|<<|«|»', ' ', text)
    
    # 3. Chuẩn hóa khoảng trắng (xóa các khoảng trắng thừa) và cắt khoảng trắng 2 đầu
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# -----------------------
# LẤY URL TỪ SITEMAP XML
# -----------------------
def get_urls_from_sitemap(max_urls=2000000):
    print("🚀 Đang thu thập danh sách URL từ Sitemap Vietnamnet...")
    urls = set()
    try:
        r = requests.get(SITEMAP_INDEX, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.content, "xml") 
        
        sitemaps = [loc.text for loc in soup.find_all("loc") if loc.text.endswith('.xml')]
        
        for sm in sitemaps:
            if len(urls) >= max_urls:
                break
                
            print(f" Đang bóc tách sitemap con: {sm}")
            try:
                sm_r = requests.get(sm, headers=HEADERS, timeout=10)
                sm_soup = BeautifulSoup(sm_r.content, "xml")
                
                for loc in sm_soup.find_all("loc"):
                    link = loc.text
                    # THAY ĐỔI: Chuyển .htm thành .html cho Vietnamnet
                    if link.endswith('.html') and not any(x in link for x in ['/video/', '/anh/', '/podcast/', '/infographic/', '/thoi-su/']):
                        urls.add(link)
                        if len(urls) >= max_urls:
                            break
            except Exception as e:
                print(f" Bỏ qua sitemap {sm} do lỗi: {e}")
                
    except Exception as e:
        print(f" Lỗi khi đọc sitemap gốc: {e}")
        
    print(f" Đã gom được {len(urls)} URLs tiềm năng để đưa vào hàng đợi.")
    return list(urls)

# -----------------------
# CRAWL ARTICLE (Sửa class HTML cho Vietnamnet)
# -----------------------
def crawl_article(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return f"Lỗi HTTP {r.status_code}", None, None

        soup = BeautifulSoup(r.content, "html.parser")

        # 0. KIỂM TRA ANTI-BOT
        page_title = soup.title.text.lower() if soup.title else ""
        if "just a moment" in page_title or "cloudflare" in page_title or "attention required" in page_title:
            return "BỊ CHẶN BỞI CLOUDFLARE/ANTI-BOT", None, None

        # 1. Lấy Summary (Sapo) - Cập nhật class của Vietnamnet
        sapo_tag = soup.find(class_="content-detail-sapo")
        if not sapo_tag:
            return "Không tìm thấy đoạn Sapo", None, None
            
        summary = clean(sapo_tag.text)
        summary_words = len(summary.split())
        
        if not (50 <= summary_words <= 110):
            return f"Sapo dài {summary_words} chữ (ngoài khoảng 50-110)", None, None

        # 2. Lấy Text gốc - Cập nhật class của Vietnamnet
        content_div = soup.find(class_="maincontent")
        if not content_div:
            return "Không tìm thấy nội dung (Sai cấu trúc HTML)", None, None

        paragraphs = content_div.find_all("p")
        if not paragraphs:
            return "Nội dung không chứa thẻ <p> hợp lệ", None, None

        text = " ".join([clean(p.text) for p in paragraphs])
        text_words = len(text.split())

        if not (400 <= text_words <= 800):
            return f"Nội dung dài {text_words} chữ (ngoài khoảng 400-800)", None, None

        # 3. TÍNH TOÁN ĐIỂM SEMANTIC SIMILARITY
        emb_summary = SIMILARITY_MODEL.encode(summary, convert_to_tensor=True)
        emb_text = SIMILARITY_MODEL.encode(text, convert_to_tensor=True)
        
        score = util.cos_sim(emb_summary, emb_text).item()

        if score < 0.7:
            return f"Similarity thấp ({score:.2f} < 0.7)", None, None

        return f"Thành công (Sim: {score:.2f})", text, summary
    except Exception as e:
        return f"Lỗi ngoại lệ: {e}", None, None

# -----------------------
# BUILD DATASET BẰNG MULTITHREADING
# -----------------------
def build():
    urls = get_urls_from_sitemap(max_urls=2000000)
    data = []
    
    print(f"\n Bắt đầu crawl đa luồng mục tiêu {TARGET_ROWS} bài...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_url = {executor.submit(crawl_article, url): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                status, text, summary = future.result()
                
                # Nếu bóc tách thành công và qua được mốc similarity
                if text and summary:
                    data.append({
                        "text": text,
                        "summary": summary
                    })
                    print(f" [NHẬN] (Đã gom: {len(data)}/{TARGET_ROWS}) | {status} | {url}")

                    # Checkpoint: Lưu file liên tục mỗi khi lấy được 100 bài
                    if len(data) % 100 == 0:
                        pd.DataFrame(data).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
                        print(f" Đã lưu tạm {len(data)} dòng vào {OUTPUT_FILE}...")

                    # Chốt chặn dừng chương trình khi đủ số lượng
                    if len(data) >= TARGET_ROWS:
                        print("🎉 Đã đủ chỉ tiêu. Đang ép dừng các luồng xử lý (có thể mất vài giây)...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                
                # Nếu bị từ chối, in ra lý do
                else:
                    print(f" [BỎ QUA] (Đã gom: {len(data)}/{TARGET_ROWS}) | Lý do: {status} | {url}")
                
            except Exception as e:
                print(f" [LỖI LUỒNG] {url} - {e}")

    # Ghi đè tệp chung cuộc
    final_df = pd.DataFrame(data[:TARGET_ROWS])
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig") 
    
    print(f"🎉 HOÀN TẤT! Đã đóng gói thành công {len(final_df)} bài vào file: {OUTPUT_FILE}")

if __name__ == "__main__":
    build()