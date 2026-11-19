# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603190

> Chỉ chứa phần bổ sung so với v1.0. CAL-01 được xác nhận qua review Part 5;
> học viên xác nhận CAL-02 đến CAL-04 trong chat ngày 2026-11-19. AI hỗ trợ soạn
> câu chữ và thực hiện QC theo yêu cầu; các quyết định calibration thuộc học viên.

## 1. Câu hỏi mở đã chốt

| Câu hỏi v1.0 | Quyết định | Trạng thái | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| #5 — Kính và phần nhìn xuyên qua kính thuộc mask nào? | Bề mặt kính thuộc mask xe. Người hoặc nội thất nhìn xuyên qua không vì thế trở thành một phần của xe; người vẫn là polygon riêng. Khoảng cabin mở lớn không lấp như kính. | Học viên đã xác nhận khi review ba mask Part 5 | img_004, img_013, img_017 | CAL-01 |
| #3 — Giỏ/thùng/container thuộc box và mask xe thế nào? | Với `three_wheeler`, giỏ/thùng/container thuộc xe. Với `bicycle` ở img_031, box/mask theo khung, bánh và tay lái, không mở rộng theo giỏ/thùng/container; cargo nằm bên trong bbox chữ nhật chỉ là overlap không tránh được. | Học viên đã xác nhận | img_003, img_031 | CAL-02 |

## 2. Ca khó mới từ lỗi máy và bẫy lọt

| Ảnh | Tình huống và bằng chứng | Quyết định / quy tắc bổ sung | Trạng thái | `case_id` |
|---|---|---|---|---|
| img_012, img_017, img_003, img_010 | YOLO hoặc bẫy sai lớp gọi xe ba bánh thành car/motorbike/bicycle/truck; profile có 19 three_wheeler deleted. | Giữ định nghĩa lớp v1.0: xe ba bánh có mui/thùng vẫn là `three_wheeler`. Thêm lượt kiểm riêng mọi proposal car/truck/motorbike/bicycle có hình xe địa phương; quyết theo kết cấu/số bánh, không theo nhãn máy. | Học viên xác nhận; đây là lỗi máy/người dưới semantic v1.0 đã rõ | CAL-03 |
| img_006, img_031 | Hai bẫy `loose` lọt; guideline đã yêu cầu box ôm phần nhìn thấy, sai số khoảng 2 px. | Không đổi định nghĩa vật. Sau lượt kiểm lớp, thực hiện lượt geometry độc lập trên 100% ảnh; không đoán phần bị che; ghi `sai_vi_tri_kich_thuoc` nếu box không ôm sát. | Học viên xác nhận | CAL-04 |

## 3. Câu hỏi còn mở

| Câu hỏi v1.0 | Trạng thái / việc cần làm |
|---|---|
| #1 — Xe đạp điện hay motorbike | Chưa có ca đủ rõ trong bằng chứng đã chọn; chưa chốt. |
| #2 — Khối xe máy đỗ dày gần camera | QC img_006 vẫn xử lý các xe còn phân biệt được như từng instance; chưa có ca đủ rõ để đặt ngưỡng chuyển sang ignore_region, nên câu hỏi tổng quát chưa chốt. |
| #3 — Hàng nhô ra | Đã chốt phạm vi theo lớp trong CAL-02; các loại cargo khác ngoài giỏ/thùng/container chưa có ví dụ rõ và tiếp tục ghi `can_review` nếu gặp. |
| #4 — Người thấp thoáng trong xe ba bánh | Chưa định lượng độ nhìn thấy nhất quán; chưa chốt. |
| #6 — Box máy chồng quá nhiều trong ảnh đêm | Chưa có quyết định calibration; chưa chốt. |
| #7 — Confidence tối thiểu để chấp nhận không sửa | Không dùng confidence thay kiểm bằng mắt; chưa chốt ngưỡng chấp nhận tự động. |

## 4. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt kính; chốt giỏ/thùng/container theo `three_wheeler` và `bicycle`; thêm lượt kiểm lớp three_wheeler và lượt geometry | CAL-01 đến CAL-04, học viên xác nhận |
