# ==============================================================
# Module: Semantic TextRank - Trích xuất câu quan trọng
# Tác giả: Viết Đức (Information Extraction & Graphs)
# Tối ưu: Level 1 (Khôi phục trật tự thời gian)
#        + Level 2 (Semantic Embedding thay TF-IDF)
# ==============================================================

# pyrefly: ignore-errors

import numpy as np
import networkx as nx
import nltk
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Tải bộ tách câu của NLTK (chỉ cần chạy 1 lần)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---- Hằng số cấu hình ----
# [Fix Bẫy #1]: Ngưỡng an toàn 700 từ (words) để bù hao cho ~1024 tokens
# Tiếng Việt bị tokenizer chẻ 1 từ thành nhiều mảnh,
# nên 700 words ≈ 1024 tokens là vùng an toàn.
SAFE_WORD_LIMIT = 700

# Tên model nhúng câu đa ngôn ngữ (hỗ trợ cả Anh và Việt, ~400MB)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Biến toàn cục để cache model (chỉ load 1 lần duy nhất)
_model = None


def _get_model():
    """Load model Sentence Transformer (lazy loading - chỉ tải khi cần)."""
    global _model
    if _model is None:
        print("[TextRank] Đang tải mô hình nhúng câu lần đầu tiên...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[TextRank] Tải mô hình thành công!")
    return _model


def _split_sentences(text: str, language: str = 'vi') -> list:
    """
    Tách văn bản thành danh sách các câu.
    - Tiếng Việt: Dùng dấu chấm câu để tách.
    - Tiếng Anh: Dùng nltk.sent_tokenize.
    Trả về: list[str] - Danh sách các câu đã lọc rác và khử trùng.
    """
    if language == 'en':
        sentences = nltk.sent_tokenize(text, language='english')
    else:
        # Tiếng Việt: nltk vẫn hoạt động khá tốt với dấu chấm câu
        sentences = nltk.sent_tokenize(text)

    # Lọc bỏ câu rác (quá ngắn, ít hơn 5 từ thường là tiêu đề hoặc chú thích)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 5]

    # [Fix Bug Trùng Lặp]: Khử trùng câu (Deduplication)
    # Giữ nguyên thứ tự xuất hiện đầu tiên, loại bỏ các bản sao phía sau.
    # Điều này chống lại trường hợp Crawler cào nhầm nội dung
    # hiển thị cho Mobile và Desktop trên cùng một trang web.
    seen = set()
    unique_sentences = []
    for s in sentences:
        # Chuẩn hóa khoảng trắng để so sánh chính xác hơn
        normalized = ' '.join(s.split())
        if normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(s)

    if len(unique_sentences) < len(sentences):
        print(f"[TextRank] Đã loại bỏ {len(sentences) - len(unique_sentences)} câu trùng lặp.")

    return unique_sentences


def _semantic_textrank(sentences: list, max_words: int = SAFE_WORD_LIMIT) -> str:
    """
    Thuật toán Semantic TextRank cốt lõi.
    Bước 1: Nhúng câu thành Vector ngữ nghĩa (Level 2 Optimization).
    Bước 2: Tính ma trận Cosine Similarity.
    Bước 3: Xây dựng đồ thị và chạy PageRank.
    Bước 4: Chọn Top N câu và sắp xếp theo thời gian (Level 1 Optimization).

    Input:  list[str] - Danh sách các câu đã tách.
    Output: str - Văn bản rút gọn dưới max_words từ.
    """
    model = _get_model()

    # ---- Bước 1: Nhúng câu thành Vector ngữ nghĩa ----
    # Mỗi câu sẽ được biến thành một mảng 384 số thực
    embeddings = model.encode(sentences, show_progress_bar=False)

    # ---- Bước 2: Tính ma trận Cosine Similarity ----
    similarity_matrix = cosine_similarity(embeddings)

    # [Fix Bẫy #2]: Triệt tiêu đường chéo (Self-loop)
    # Nếu không làm bước này, mỗi câu sẽ tự bầu cho chính mình
    # với điểm 1.0, làm sai lệch kết quả PageRank.
    np.fill_diagonal(similarity_matrix, 0)

    # ---- Bước 3: Xây dựng đồ thị và chạy PageRank ----
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph, max_iter=100)

    # ---- Bước 4: Phân cụm ngữ nghĩa (K-Means) và Chọn xoay vòng ----
    if len(sentences) >= 5:
        # Số lượng cụm K: tối đa 6, tối thiểu 2
        num_clusters = min(6, max(2, len(sentences) // 5))
        
        # Phân cụm các vector câu (embeddings)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Nhóm index theo cụm
        cluster_to_indices = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            cluster_to_indices[label].append(idx)
            
        # Sắp xếp câu trong từng cụm theo PageRank
        for label in cluster_to_indices:
            cluster_to_indices[label].sort(key=lambda x: scores.get(x, 0), reverse=True)
            
        print(f"[TextRank] Đã phân thành {num_clusters} cụm chủ đề ngữ nghĩa.")
        
        # Xoay vòng lấy câu từ các cụm
        ranked_indices = []
        cluster_pointers = {label: 0 for label in cluster_to_indices}
        active_clusters = list(cluster_to_indices.keys())
        
        while active_clusters:
            for label in list(active_clusters):
                pointer = cluster_pointers[label]
                if pointer < len(cluster_to_indices[label]):
                    ranked_indices.append(cluster_to_indices[label][pointer])
                    cluster_pointers[label] += 1
                else:
                    active_clusters.remove(label)
    else:
        # Ít câu thì sắp xếp thẳng theo PageRank
        ranked_indices = sorted(scores, key=scores.get, reverse=True)

    # ---- Bước 5: Chọn Top N câu (đảm bảo <= max_words từ) ----
    selected_indices = []
    current_word_count = 0

    for idx in ranked_indices:
        sentence_word_count = len(sentences[idx].split())
        if current_word_count + sentence_word_count <= max_words:
            selected_indices.append(idx)
            current_word_count += sentence_word_count
        # Nếu thêm câu này sẽ vượt ngưỡng -> dừng lại
        if current_word_count >= max_words:
            break

    # ---- [Level 1 Optimization]: Khôi phục trật tự thời gian ----
    # Sắp xếp lại theo index gốc (thứ tự xuất hiện trong bài báo)
    # để ViT5 đọc được văn bản mạch lạc: Mở bài -> Thân bài -> Kết bài
    selected_indices.sort()

    # Ghép các câu đã chọn thành một chuỗi String duy nhất
    result = " ".join([sentences[i] for i in selected_indices])
    return result


def process_textrank(raw_text: str, language: str = 'vi', max_words: int = SAFE_WORD_LIMIT) -> str:
    """
    Hàm chính (Entry Point) - Cơ chế Định tuyến theo độ dài.

    - Input:  raw_text (str)  - Bài báo thô.
              language (str)  - 'vi' hoặc 'en'.
              max_words (int) - Ngưỡng an toàn (mặc định 700 từ).
    - Output: str - Văn bản sẵn sàng đưa vào ViT5.

    Trường hợp 1: Bài báo ngắn (< 700 từ) -> Trả về nguyên xi.
    Trường hợp 2: Bài báo dài (>= 700 từ)  -> Kích hoạt Semantic TextRank.
    """
    word_count = len(raw_text.split())

    # --- Trường hợp 1: Bài báo ngắn -> Không cần TextRank ---
    if word_count < max_words:
        print(f"[TextRank] Bài báo ngắn ({word_count} từ). Bỏ qua TextRank.")
        return raw_text

    # --- Trường hợp 2: Bài báo dài -> Kích hoạt Semantic TextRank ---
    print(f"[TextRank] Bài báo dài ({word_count} từ). Kích hoạt Semantic TextRank...")
    sentences = _split_sentences(raw_text, language)

    # Nếu sau khi tách câu chỉ còn <= 3 câu thì không đáng để chạy thuật toán
    if len(sentences) <= 3:
        return raw_text

    result = _semantic_textrank(sentences, max_words)
    print(f"[TextRank] Đã nén xuống còn ~{len(result.split())} từ.")
    return result


# ==============================================================
# Khối chạy thử nghiệm (Test) - Chạy trực tiếp file này để kiểm tra
# ==============================================================
if __name__ == "__main__":
    # Bài báo mẫu dài (nội dung thực, KHÔNG nhân bản, > 700 từ)
    long_article = """
Hà Nội, ngày 8 tháng 6 năm 2026. Thủ tướng Chính phủ vừa ban hành nghị định mới
về phát triển trí tuệ nhân tạo tại Việt Nam. Theo đó, các trường đại học sẽ được
khuyến khích đưa môn học AI vào chương trình đào tạo bắt buộc từ năm học 2027.
Bộ Giáo dục và Đào tạo cho biết, hiện nay có hơn 50 trường đại học trên cả nước
đã triển khai giảng dạy các môn liên quan đến trí tuệ nhân tạo và khoa học dữ liệu.
Tuy nhiên, chất lượng đào tạo vẫn còn nhiều bất cập do thiếu giảng viên có kinh nghiệm
thực tế và cơ sở vật chất chưa đáp ứng được yêu cầu. Đại diện Google Việt Nam cho biết
công ty sẵn sàng hỗ trợ đào tạo giảng viên và cung cấp tài liệu học tập miễn phí.
Trong khi đó, tập đoàn FPT đã đầu tư hơn 1000 tỷ đồng vào nghiên cứu và phát triển AI.
Ông Trương Gia Bình, Chủ tịch FPT, nhấn mạnh rằng Việt Nam có tiềm năng trở thành
trung tâm AI của khu vực Đông Nam Á nếu có chính sách đúng đắn. Các chuyên gia quốc tế
cũng đánh giá cao tiềm năng phát triển AI của Việt Nam nhờ lực lượng lao động trẻ và
năng động. Tuy nhiên, thách thức lớn nhất vẫn là vấn đề bảo mật dữ liệu cá nhân và
quy định pháp lý chưa theo kịp tốc độ phát triển của công nghệ. Bộ Công an đang soạn
thảo khung pháp lý mới để quản lý việc sử dụng AI trong các lĩnh vực nhạy cảm như
nhận dạng khuôn mặt và giám sát an ninh. Nghị định mới dự kiến sẽ có hiệu lực từ
ngày 1 tháng 1 năm 2027, mở ra một kỷ nguyên mới cho ngành công nghệ thông tin
tại Việt Nam. Các doanh nghiệp công nghệ trong nước đang rất kỳ vọng vào những
thay đổi tích cực từ chính sách này.

Bên cạnh đó, Bộ Khoa học và Công nghệ cũng công bố chiến lược quốc gia về AI giai đoạn
2026-2030. Chiến lược này đặt mục tiêu đưa Việt Nam vào top 5 quốc gia dẫn đầu ASEAN
về nghiên cứu và ứng dụng trí tuệ nhân tạo. Ngân sách dành cho nghiên cứu AI sẽ được
tăng gấp 3 lần so với giai đoạn trước, lên mức 15.000 tỷ đồng trong 5 năm tới.
Các viện nghiên cứu lớn như Viện Hàn lâm Khoa học và Công nghệ Việt Nam, Đại học
Bách khoa Hà Nội và Đại học Quốc gia TP.HCM sẽ là những đơn vị nòng cốt thực hiện
chiến lược này. Giáo sư Nguyễn Thanh Thủy, Phó Hiệu trưởng Đại học Công nghệ,
nhận định rằng việc đào tạo nguồn nhân lực chất lượng cao là yếu tố then chốt quyết
định sự thành công của chiến lược. Ông cho biết trường đã mở thêm 3 chương trình đào
tạo thạc sĩ chuyên ngành AI, Machine Learning và Data Science từ năm 2025.

Về phía doanh nghiệp, VinAI Research tiếp tục khẳng định vị thế là phòng nghiên cứu
AI hàng đầu Đông Nam Á với hơn 50 bài báo được chấp nhận tại các hội nghị quốc tế
uy tín như NeurIPS, ICML và CVPR trong năm 2025. Tiến sĩ Bùi Hải Hưng, Viện trưởng
VinAI, chia sẻ rằng đội ngũ đang tập trung phát triển các mô hình ngôn ngữ lớn
cho tiếng Việt, nhằm tạo ra một phiên bản GPT bản địa phục vụ cho các doanh nghiệp
và tổ chức trong nước. Ngoài ra, công ty cũng đang hợp tác với nhiều bệnh viện lớn
để ứng dụng AI trong chẩn đoán hình ảnh y tế, giúp phát hiện sớm các bệnh ung thư
và tim mạch với độ chính xác lên tới 95 phần trăm.

Trong lĩnh vực nông nghiệp, nhiều tỉnh thành đã bắt đầu triển khai các giải pháp AI
để tối ưu hóa quy trình canh tác. Tỉnh Lâm Đồng là một trong những địa phương tiên
phong ứng dụng drone và AI để giám sát sức khỏe cây trồng trên diện tích hơn 5.000
héc-ta cà phê. Hệ thống AI có thể phân tích ảnh chụp từ drone để phát hiện sớm các
dấu hiệu sâu bệnh, thiếu nước hoặc thiếu dinh dưỡng, giúp nông dân tiết kiệm tới
30 phần trăm chi phí phân bón và thuốc trừ sâu. Ông Phạm S, Phó Chủ tịch UBND tỉnh
Lâm Đồng, cho biết mô hình này sẽ được nhân rộng ra toàn tỉnh trong năm 2027.

Tại TP.HCM, Khu Công nghệ cao đã khánh thành Trung tâm Đổi mới sáng tạo về AI
với tổng vốn đầu tư 500 tỷ đồng. Trung tâm này được trang bị hệ thống máy chủ GPU
hiện đại nhất Đông Nam Á, bao gồm 100 card NVIDIA H100, cho phép các startup và
nhà nghiên cứu huấn luyện các mô hình AI quy mô lớn mà không cần thuê dịch vụ
đám mây nước ngoài. Bà Lê Bích Loan, Phó Trưởng ban quản lý Khu CNC, kỳ vọng
trung tâm sẽ thu hút ít nhất 50 startup AI trong năm đầu tiên hoạt động.

Cuối cùng, về mặt pháp lý, Quốc hội đang xem xét dự thảo Luật Trí tuệ nhân tạo,
dự kiến sẽ được thông qua vào kỳ họp tháng 10 năm 2026. Luật này sẽ quy định rõ
trách nhiệm pháp lý khi AI gây ra thiệt hại, bảo vệ quyền riêng tư của công dân
trước các hệ thống giám sát tự động, và thiết lập khung quản lý cho các ứng dụng
AI trong lĩnh vực tài chính ngân hàng. Đại biểu Quốc hội Nguyễn Thị Kim Ngân
nhấn mạnh rằng luật cần phải cân bằng giữa việc thúc đẩy đổi mới sáng tạo và
bảo vệ quyền lợi của người dân trước những rủi ro tiềm ẩn từ công nghệ AI.
    """.strip()

    # --- Test 1: Bài báo dài, nội dung KHÔNG trùng lặp ---
    print("=" * 60)
    print("TEST 1: Bài báo dài (nội dung thực, không trùng lặp)")
    print(f"Độ dài bài báo gốc: {len(long_article.split())} từ")
    print("=" * 60)

    result = process_textrank(long_article, language='vi')

    print("\n" + "=" * 60)
    print("KẾT QUẢ SAU KHI TEXTRANK NÉN:")
    print("=" * 60)
    print(result)
    print(f"\nĐộ dài sau nén: {len(result.split())} từ")

    # --- Test 2: Bài báo có nội dung trùng lặp (giả lập Crawler lỗi) ---
    print("\n\n" + "=" * 60)
    print("TEST 2: Bài báo bị Crawler cào trùng lặp (x3)")
    duplicated_article = " ".join([long_article] * 3)
    print(f"Độ dài trước khử trùng: {len(duplicated_article.split())} từ")
    print("=" * 60)

    result2 = process_textrank(duplicated_article, language='vi')
    print(f"\nĐộ dài sau nén: {len(result2.split())} từ")
