# pyrefly: ignore-errors
# ==============================================================
# Module: Knowledge Graph - Trích xuất thực thể & Đồ thị Tri thức
# Tác giả: Viết Đức (Information Extraction & Graphs)
# Mô hình: Davlan/bert-base-multilingual-cased-ner-hrl (đa ngôn ngữ)
# Fallback: underthesea (khi Transformers lỗi hoặc thiếu RAM)
# ==============================================================

import re
import nltk
import networkx as nx
from itertools import combinations
from collections import defaultdict

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---- Hằng số cấu hình ----
# Tên mô hình NER đa ngôn ngữ (hỗ trợ cả Anh, Việt, và nhiều ngôn ngữ khác)
NER_MODEL_NAME = "Davlan/bert-base-multilingual-cased-ner-hrl"

# Các nhãn thực thể mà chúng ta quan tâm
VALID_LABELS = {"PER", "ORG", "LOC"}

# Bảng màu cho từng loại thực thể (dùng cho Pyvis)
LABEL_COLORS = {
    "PER": "#3498db",   # Xanh dương - Người (Person)
    "ORG": "#e67e22",   # Cam - Tổ chức (Organization)
    "LOC": "#f1c40f",   # Vàng hổ phách - Địa điểm (Location)
}

# Biến toàn cục để cache pipeline NER (Lazy Loading - chỉ tải 1 lần)
_ner_pipeline = None
_use_fallback = False  # Cờ đánh dấu: True nếu Transformers lỗi, chuyển sang underthesea


def _get_ner_pipeline():
    """
    Tải pipeline NER (Lazy Loading).
    """
    global _ner_pipeline, _use_fallback

    if _ner_pipeline is not None:
        return _ner_pipeline

    if _use_fallback:
        return None

    try:
        from transformers import pipeline
        print("[NER] Đang tải mô hình Davlan (lần đầu, ~700MB)...")
        _ner_pipeline = pipeline(
            "ner",
            model=NER_MODEL_NAME,
            aggregation_strategy="simple"  # Gộp subwords: "Hà" + "##Nội" → "Hà Nội"
        )
        print("[NER] Tải mô hình thành công!")
        return _ner_pipeline
    except Exception as e:
        print(f"[NER] ⚠️ Lỗi tải mô hình Transformers: {e}")
        print("[NER] → Chuyển sang dùng Underthesea (Fallback)...")
        _use_fallback = True
        return None


def _extract_with_transformers(sentence, pipe):
    """
    Trích xuất thực thể từ 1 câu bằng pipeline Transformers.
    """
    results = []
    try:
        raw_entities = pipe(sentence)
        for ent in raw_entities:
            label = ent.get("entity_group", "").upper()
            word = ent.get("word", "").strip()

            if label in VALID_LABELS and len(word) > 1:
                results.append({"entity": word, "label": label})
    except Exception as e:
        print(f"[NER] ⚠️ Lỗi khi xử lý câu: {e}")

    return results


def _extract_with_underthesea(sentence):
    """
    Trích xuất thực thể từ 1 câu bằng thư viện Underthesea (Fallback).
    """
    results = []
    try:
        from underthesea import ner as uts_ner
        raw = uts_ner(sentence)

        current_entity = ""
        current_label = ""

        for token_tuple in raw:
            word = token_tuple[0]
            ner_tag = token_tuple[3] if len(token_tuple) > 3 else "O"

            if ner_tag.startswith("B-"):
                if current_entity and current_label in VALID_LABELS:
                    results.append({"entity": current_entity.strip(), "label": current_label})
                current_label = ner_tag[2:]  # "B-PER" → "PER"
                current_entity = word
            elif ner_tag.startswith("I-") and ner_tag[2:] == current_label:
                current_entity += " " + word
            else:
                if current_entity and current_label in VALID_LABELS:
                    results.append({"entity": current_entity.strip(), "label": current_label})
                current_entity = ""
                current_label = ""

        if current_entity and current_label in VALID_LABELS:
            results.append({"entity": current_entity.strip(), "label": current_label})

    except Exception as e:
        print(f"[NER] ⚠️ Lỗi Underthesea: {e}")

    return results


def split_vietnamese_sentences(text: str) -> list:
    """
    Tách văn bản thành các câu độc lập, có xử lý tránh viết tắt tiếng Việt
    (ví dụ: 'TP. Hồ Chí Minh', 'ThS. Nguyễn Văn A', 'GS. TS. Lê Văn B').
    """
    # Thay thế tạm thời dấu chấm sau các từ viết tắt phổ biến
    abbreviations = [
        r"\bTP\b", r"\bThS\b", r"\bGS\b", r"\bTS\b", r"\bNXB\b",
        r"\bThứ\s+[2-7]\b", r"\bCN\b", r"\bđ/c\b", r"\bV/v\b",
        r"\bP\b", r"\bQ\b", r"\bH\b", r"\bTỉnh\b", r"\bMr\b", r"\bMrs\b"
    ]
    
    modified_text = text
    # Tìm và thay thế tạm thời "TP." -> "TP<DOT>" để NLTK không cắt nhầm câu
    for abbr in abbreviations:
        modified_text = re.sub(abbr + r"\.", lambda m: m.group(0)[:-1] + "___DOT___", modified_text, flags=re.IGNORECASE)
    
    # Sử dụng NLTK để tách câu
    raw_sentences = nltk.sent_tokenize(modified_text)
    
    # Khôi phục lại dấu chấm gốc
    final_sentences = []
    for sent in raw_sentences:
        restored = sent.replace("___DOT___", ".")
        final_sentences.append(restored)
        
    return final_sentences


# ==============================================================
# HÀM CHÍNH 1: Trích xuất thực thể từ văn bản
# ==============================================================
def extract_entities(text: str, language: str = 'vi') -> list:
    """
    Trích xuất thực thể (NER) từ văn bản.
    - Cơ chế Chunking: Cắt văn bản thành từng câu để tránh lỗi tràn 512 tokens.
    - Xử lý thông minh dấu chấm viết tắt tiếng Việt.
    - Input: Văn bản (str), ngôn ngữ (str: 'vi' hoặc 'en')
    - Output: Danh sách thực thể kèm sentence_id và nội dung câu gốc.
      VD: [{"entity": "FPT", "label": "ORG", "sentence_id": 0, "sentence_text": "..."}, ...]
    """
    if not text or not text.strip():
        return []

    # ---- Bước 1: Chunking - Cắt văn bản thành từng câu thông minh ----
    sentences = split_vietnamese_sentences(text)
    print(f"[NER] Đã tách văn bản thành {len(sentences)} câu.")

    all_entities = []

    # ---- Bước 2: Chạy NER trên từng câu ----
    pipe = _get_ner_pipeline()

    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        if pipe is not None and not _use_fallback:
            entities = _extract_with_transformers(sentence, pipe)
        else:
            entities = _extract_with_underthesea(sentence)

        # Gắn thông tin ngữ cảnh để dùng cho việc vẽ quan hệ trên đồ thị
        for ent in entities:
            ent["sentence_id"] = idx
            ent["sentence_text"] = sentence
            all_entities.append(ent)

    print(f"[NER] Tổng số thực thể tìm được: {len(all_entities)}")
    return all_entities


# ==============================================================
# HÀM CHÍNH 2: Xây dựng Đồ thị Tri thức
# ==============================================================
def build_knowledge_graph(entities: list, output_path: str = "graph.html"):
    """
    Vẽ đồ thị tri thức (Knowledge Graph) từ danh sách thực thể.
    - Thuật toán: Co-occurrence (đồng xuất hiện trong cùng 1 câu).
    - Cải tiến 1: Hiển thị ngữ cảnh (câu văn gốc chứa 2 thực thể) khi hover chuột vào đường nối.
    - Cải tiến 2: Thêm Bảng chú giải (Legend) ở góc màn hình.
    - Input: Danh sách thực thể từ hàm extract_entities().
    - Output: File HTML tương tác (graph.html).
    """
    if not entities:
        print("[KG] Không có thực thể nào để vẽ đồ thị.")
        return

    # ---- Bước 1: Khởi tạo đồ thị NetworkX ----
    G = nx.Graph()

    # ---- Bước 2: Thêm Nodes (Loại bỏ trùng lặp) ----
    unique_entities = {}
    for ent in entities:
        name = ent["entity"]
        label = ent["label"]
        if name not in unique_entities:
            unique_entities[name] = label

    for name, label in unique_entities.items():
        color = LABEL_COLORS.get(label, "#95a5a6")
        G.add_node(
            name,
            label=name,
            color=color,
            title=f"Thực thể: {name}<br>Phân loại: {label}",  # Tooltip khi hover node
            group=label
        )

    print(f"[KG] Đã thêm {len(unique_entities)} nodes vào đồ thị.")

    # ---- Bước 3: Thêm Edges (Co-occurrence + Trích xuất ngữ cảnh) ----
    # Nhóm thực thể và câu văn theo sentence_id
    sentence_groups = defaultdict(list)
    sentence_texts = {}
    for ent in entities:
        sentence_groups[ent["sentence_id"]].append(ent["entity"])
        sentence_texts[ent["sentence_id"]] = ent["sentence_text"]

    # Lưu ngữ cảnh cho từng cặp thực thể
    # key: (e1, e2), value: set của các câu văn chứa cặp này
    edge_contexts = defaultdict(set)

    for sid, entity_names in sentence_groups.items():
        unique_in_sentence = list(set(entity_names))
        if len(unique_in_sentence) < 2:
            continue

        sentence_str = sentence_texts[sid]

        # Tạo tất cả các cặp thực thể trong cùng 1 câu
        for e1, e2 in combinations(unique_in_sentence, 2):
            # Đảm bảo thứ tự để không bị trùng (e1, e2) và (e2, e1)
            pair = tuple(sorted([e1, e2]))
            edge_contexts[pair].add(sentence_str)

    # Thêm các cạnh vào đồ thị NetworkX cùng với ngữ cảnh và trọng số
    edge_count = 0
    for (e1, e2), contexts in edge_contexts.items():
        weight = len(contexts)
        # Tạo nhãn tooltip cho đường nối: gom các câu chứa chúng lại
        tooltip_lines = []
        for idx, ctx in enumerate(contexts):
            # Bôi đậm 2 thực thể trong câu để dễ nhìn
            highlighted = ctx.replace(e1, f"<b>{e1}</b>").replace(e2, f"<b>{e2}</b>")
            tooltip_lines.append(f"{idx+1}. ... {highlighted} ...")
            
        tooltip = "<b>Ngữ cảnh xuất hiện:</b><br>" + "<br>".join(tooltip_lines)

        G.add_edge(e1, e2, weight=weight, title=tooltip)
        edge_count += 1

    print(f"[KG] Đã thêm {edge_count} cạnh (edges) vào đồ thị.")

    # ---- Bước 4: Render đồ thị bằng Pyvis ----
    try:
        from pyvis.network import Network

        net = Network(
            height="720px",
            width="100%",
            bgcolor="#1a1a2e",     # Nền tối (Dark mode) sang trọng
            font_color="white",
            notebook=False
        )

        net.from_nx(G)

        # Cấu hình physics layout và thuộc tính tương tác
        net.set_options("""
        {
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "font": {
                    "size": 16,
                    "face": "Arial",
                    "color": "#ffffff"
                },
                "shadow": {
                    "enabled": true
                }
            },
            "edges": {
                "color": {
                    "inherit": "both",
                    "opacity": 0.6
                },
                "smooth": {
                    "type": "continuous"
                }
            },
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -80,
                    "centralGravity": 0.005,
                    "springLength": 200,
                    "springConstant": 0.05
                },
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "iterations": 200
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "zoomView": true,
                "dragView": true
            }
        }
        """)

        # Tối ưu kích thước node dựa trên mức độ liên kết (degree)
        for node in net.nodes:
            node_name = node["id"]
            degree = G.degree(node_name)
            node["size"] = 18 + degree * 4

        # Tối ưu độ dày của cạnh theo tần suất co-occurrence
        for edge in net.edges:
            e1, e2 = edge["from"], edge["to"]
            weight = 1
            if G.has_edge(e1, e2):
                weight = G[e1][e2].get("weight", 1)
            edge["width"] = 1.5 + weight * 1.5

        # ---- Cải tiến 2: Thêm Bảng chú giải (Legend) góc đồ thị bằng HTML chèn thêm ----
        legend_html = f"""
        <div style="
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(26, 26, 46, 0.9);
            border: 2px solid #3e3e5e;
            border-radius: 8px;
            padding: 15px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #ffffff;
            z-index: 999;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            pointer-events: none;
        ">
            <h4 style="margin: 0 0 10px 0; border-bottom: 1px solid #3e3e5e; padding-bottom: 5px; color: #00e6ff;">
                Đồ Thị Tri Thức
            </h4>
            <div style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="display: inline-block; width: 15px; height: 15px; background: {LABEL_COLORS['PER']}; border-radius: 50%; margin-right: 10px;"></span>
                <span>Người (PER)</span>
            </div>
            <div style="margin-bottom: 8px; display: flex; align-items: center;">
                <span style="display: inline-block; width: 15px; height: 15px; background: {LABEL_COLORS['ORG']}; border-radius: 50%; margin-right: 10px;"></span>
                <span>Tổ chức (ORG)</span>
            </div>
            <div style="margin-bottom: 12px; display: flex; align-items: center;">
                <span style="display: inline-block; width: 15px; height: 15px; background: {LABEL_COLORS['LOC']}; border-radius: 50%; margin-right: 10px;"></span>
                <span>Địa điểm (LOC)</span>
            </div>
            <div style="font-size: 12px; color: #8a8ab0; border-top: 1px solid #3e3e5e; padding-top: 5px; line-height: 1.4;">
                • Kích thước nút = số kết nối<br>
                • Độ dày cạnh = tần suất xuất hiện chung<br>
                • Rê chuột vào cạnh để xem câu văn ngữ cảnh!
            </div>
        </div>
        """
        
        # Lưu file HTML
        net.save_graph(output_path)
        
        # Chèn bảng Legend vào trong file HTML vừa xuất ra
        with open(output_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        # Chèn thẻ Legend HTML vào ngay trước </body>
        new_html_content = html_content.replace("</body>", f"{legend_html}\n</body>")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(new_html_content)

        print(f"[KG] ✅ Đã lưu đồ thị cải tiến vào file: {output_path}")
        print(f"[KG] → Mở file này bằng trình duyệt để xem đồ thị tương tác!")

    except ImportError:
        print("[KG] ⚠️ Chưa cài thư viện pyvis. Chạy: pip install pyvis")
    except Exception as e:
        print(f"[KG] ⚠️ Lỗi khi render đồ thị: {e}")


# ==============================================================
# Khối chạy thử nghiệm (Test) - Chạy trực tiếp file này để kiểm tra
# ==============================================================
if __name__ == "__main__":
    # Bài báo mẫu chứa nhiều thực thể (Người, Tổ chức, Địa điểm)
    sample_text = """
    Hà Nội, ngày 8 tháng 6 năm 2026. Thủ tướng Chính phủ vừa ban hành nghị định mới
    về phát triển trí tuệ nhân tạo tại Việt Nam. Ông Trương Gia Bình, Chủ tịch FPT,
    nhấn mạnh rằng Việt Nam có tiềm năng trở thành trung tâm AI của khu vực Đông Nam Á.
 
    Đại diện Google Việt Nam cho biết công ty sẵn sàng hỗ trợ đào tạo giảng viên.
    Tại TP. Hồ Chí Minh, Đại học Bách khoa đã ký kết hợp tác với Samsung
    để xây dựng phòng thí nghiệm AI hiện đại. Giáo sư Nguyễn Thanh Thủy từ
    Đại học Quốc gia Hà Nội đánh giá rất cao chương trình này.
 
    Tập đoàn Vingroup cũng công bố đầu tư 500 tỷ đồng vào nghiên cứu AI.
    Ông Phạm Nhật Vượng khẳng định Vingroup sẽ hợp tác chặt chẽ với FPT
    và Google để đào tạo 10.000 kỹ sư AI trong 3 năm tới.
    """

    print("=" * 60)
    print("TEST: Trích xuất thực thể và vẽ Đồ thị Tri thức CẢI TIẾN")
    print("=" * 60)

    # Bước 1: Trích xuất thực thể
    entities = extract_entities(sample_text, language='vi')

    print("\n--- Danh sách thực thể tìm được ---")
    for ent in entities:
        print(f"  [{ent['label']}] {ent['entity']} (câu {ent['sentence_id']})")

    # Bước 2: Vẽ đồ thị tri thức
    print("\n--- Xây dựng đồ thị ---")
    build_knowledge_graph(entities, output_path="data/knowledge_graph.html")

