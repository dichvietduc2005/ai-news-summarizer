# Giao việc cho Thành viên 2

def extract_entities(text: str, language: str = 'vi') -> list:
    """
    Trích xuất thực thể (NER).
    - Input: Văn bản (str)
    - Output: Danh sách các thực thể. VD: [{'entity': 'Tim Cook', 'label': 'PER'}]
    """
    # TODO: TV2 load mô hình NlpHUST cho 'vi' và dslim cho 'en' (hoặc underthesea)
    pass

def build_knowledge_graph(entities: list, output_path: str = "graph.html"):
    """
    Vẽ đồ thị tri thức từ danh sách thực thể và lưu thành file HTML.
    """
    # TODO: TV2 sử dụng thư viện pyvis hoặc networkx
    pass
