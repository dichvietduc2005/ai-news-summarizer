import sys
import os
import io

# Fix encoding error on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.summarizer import generate_summary

sample_text = """Đột phá trí tuệ nhân tạo: Ứng dụng AI vào quản lý giao thông giúp giảm 30% tình trạng kẹt xe tại TP.HCM
Sáng ngày 2/7/2026, Sở Giao thông Vận tải TP.HCM phối hợp cùng một tập đoàn công nghệ lớn đã chính thức công bố kết quả thử nghiệm hệ thống quản lý giao thông thông minh ứng dụng Trí tuệ Nhân tạo (AI). Sau 6 tháng triển khai thí điểm tại các điểm nóng kẹt xe như ngã tư Hàng Xanh và khu vực vòng xoay Lăng Cha Cả, kết quả cho thấy tình trạng ùn tắc đã giảm đáng kể.

Hệ thống AI này hoạt động dựa trên việc thu thập dữ liệu theo thời gian thực từ hàng ngàn camera an ninh và cảm biến trên đường phố. Thay vì sử dụng bộ đếm thời gian cố định cho đèn tín hiệu, AI sẽ tự động phân tích mật độ xe cộ đang lưu thông và điều chỉnh thời lượng đèn xanh, đèn đỏ sao cho luồng giao thông được lưu thoát tối ưu nhất.

Theo báo cáo dự án, thời gian chờ đợi trung bình của người dân vào các khung giờ cao điểm đã giảm khoảng 30%, đồng thời tốc độ di chuyển trung bình của các phương tiện tăng 15%. Sự cải thiện này không chỉ giúp người dân tiết kiệm thời gian di chuyển mà còn góp phần giảm thiểu một lượng lớn khí thải carbon từ các phương tiện giao thông do phải nổ máy chờ đợi quá lâu.

Phát biểu tại buổi họp báo, chuyên gia công nghệ Lê Văn A nhận định: "Đây là bước tiến quan trọng trong việc định hình và xây dựng thành phố thông minh. Việc để máy học tự động ra quyết định điều tiết giao thông mang lại độ trễ thấp và tính hiệu quả vượt trội so với con người điều hành thủ công, đặc biệt là trong những tình huống giao thông diễn biến phức tạp".

Dự kiến trong năm 2027, hệ thống quản lý thông minh này sẽ được mở rộng triển khai ra toàn bộ các quận trung tâm và tích hợp thêm tính năng tự động cảnh báo, dự báo các điểm ngập úng nguy hiểm vào mùa mưa để người dân chủ động thay đổi lộ trình."""

try:
    summary = generate_summary(sample_text, language='vi')
    print("\n✨ BẢN TÓM TẮT THỰC TẾ TỪ DPO:")
    print("-" * 50)
    print(summary)
    print("-" * 50)
except Exception as e:
    print(f"Lỗi khi chạy mô hình: {e}")
