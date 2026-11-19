# Báo cáo Day 08 — 2A202603190 - Nguyễn Hải Đăng

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | person 0.15; motorbike 0.10; bicycle / car / bus / truck 0.30 | r01/r02 trong `model_log.csv`; ngưỡng đã được chấp nhận. Theo `qc/threshold_sweep.csv`, recall person 22/24 = 91.7%, motorbike 10/11 = 90.9%. Bicycle có 4 vật, bus 0, truck 1 nên dùng mặc định; car chỉ đạt 7/8 = 87.5%, chưa đạt mục tiêu 90%. |
| three_wheeler | YOLOE-26s, prompt: tuk-tuk / cyclo / cargo tricycle | 0.15 | r03 trong `model_log.csv`; ngưỡng đã được chấp nhận. Theo `qc/threshold_sweep.csv`, recall 7/7 = 100.0%, precision 7/9 = 77.8%; mẫu còn nhỏ. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.15,motorbike=0.1,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 base (`sam2.1_b.pt`), checkpoint có sẵn, chạy CUDA trên GPU 6 GB, phù hợp phần cứng máy, không cần quyền gated của SAM 3.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Xe ba bánh bị gán car/motorbike/bicycle/truck cần kiểm và sửa lớp theo kết cấu/số bánh, không chỉ đổi confidence; xem CAL-03 trong `qc/calibration_decisions.csv`. Bẫy sai lớp còn lọt ở img_003 và img_010 theo `qc/review_stats.json`. Cần sửa lớp và kiểm geometry trên toàn batch; chưa có `labels/qc_fixed.json`, nên chưa ghi nhận hoàn tất sửa nhãn.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): img_036.jpg, YOLO26 prediction #1652 (`motorbike`) có confidence 0.1155 nhưng khớp gold #43 cùng lớp, IoU 0.9637; đây là ví dụ điểm thấp vẫn đúng theo tiêu chí ghép gold (IoU ≥0.5), không có nghĩa mọi chi tiết box đều đạt guideline. Ngưỡng motorbike 0.10 giữ được vật này, còn 0.25 sẽ loại nó. Nguồn: `qc/threshold_comparison.json`.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: kết quả in-sample trên 4 ảnh không đại diện chất lượng 24 ảnh batch. `qc/review_stats.json` dùng 55 đối tượng gold; tham chiếu còn có sửa đổi hậu kiểm được nêu ở mục 8.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 591 | 81.5% | 725 box máy sau loại vùng bẫy, theo `qc/review_stats.json` |
| edited | 17 | 2.3% | 725 box máy; 13 box không ghép trong ignore_region không tính deleted, nên ba trạng thái máy không cộng đủ 725 |
| deleted | 104 | 14.3% | 725 box máy |
| added | 31 | 4.9% | 639 đối tượng sau duyệt, theo `qc/review_stats.json` |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? Theo tỉ lệ deleted trong các box có trạng thái accepted/edited/deleted, truck tệ nhất: 5/(1+0+5) = 83.3%; có thêm 3 truck phải bổ sung. QC img_003 cho thấy truck #2/#3 chỉ bao một phần xe, nên nhận đúng lớp chưa đủ để chấp nhận geometry. Three_wheeler cũng yếu: 19/(18+1+19) = 50.0% bị xóa và 4 vật được thêm; bẫy sai lớp còn lọt ở img_003/img_010. Đây là chỉ báo khối lượng sửa, không phải precision độc lập; mẫu truck rất nhỏ và nhãn sau duyệt vẫn còn lỗi. Nguồn: `qc/review_stats.json`, `qc/qc_review.csv`.
- So với gold: precision / recall của **máy** 48/82 = 58.5% / 48/55 = 87.3%, theo `qc/review_stats.json` (mẫu số: 55 đối tượng gold trên 4 ảnh; precision dùng 82 dự đoán được tính)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 6/12 = 50.0% ở lượt đầu (`qc/review_stats.json`); loại nào lọt: extra 1/3 (img_008), wrong_class 2/3 (img_003, img_010), missing 1/3 (img_005), loose 2/3 (img_031, img_006); vì sao (bạn đã bỏ bước nào của checklist)? Label các `ignore_region` cho các vùng xe bị dính quá đông và quá sát, thay vào đó đã tin máy chỉ sửa box/class.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Accepted 591/725 = 81.5% chỉ cho biết nhiều proposal được giữ gần nguyên, không chứng minh chúng đúng. Bắt bẫy chỉ 6/12 = 50.0%, kèm 5/5 ảnh QC có lỗi, cho thấy lượt duyệt chưa đủ độc lập với gợi ý máy; phù hợp với việc đã tin máy và tập trung sửa box/class thay vì đối chiếu đầy đủ guideline như ghi ở trên. Cần tách lượt tìm vật khi ẩn nhãn khỏi lượt kiểm lớp, geometry và điều kiện ignore_region. Nguồn: `qc/review_stats.json`, `qc/qc_report_auto.md`. Đây là phân tích dấu hiệu thiên kiến, không phải phép đo trực tiếp tâm lý người duyệt.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 60/4 = 15.00 phút/ảnh vs duyệt 74/24 = 3.08 phút/ảnh (cả hai thời gian active do học viên xác nhận; duyệt chưa gồm rework), tương đương nhanh hơn khoảng 4.86 lần trước khi tính rework. Thiên lệch (bạn gán gold trước nên đã quen cảnh): gold có thể giúp quen cách hiểu guideline và loại cảnh, nên so sánh thời gian không phải thử nghiệm độc lập; gold 4 ảnh và batch 24 ảnh khác nhau về độ khó/mật độ vật. Vì batch chưa sửa xong, 4.86 lần là tốc độ lượt đầu, không phải speedup của nhãn đạt chuẩn cuối cùng.
- Kết luận: chỉ dùng máy để tạo proposal có kiểm tra, chưa dùng để chấp nhận nhãn hoặc kết luận tiết kiệm thời gian, dựa trên recall person 22/24 = 91.7%, motorbike 10/11 = 90.9%, three_wheeler 7/7 = 100.0% trên gold nhưng precision máy chung chỉ 48/82 = 58.5%; lượt duyệt đầu bắt 6/12 bẫy và QC còn 5/5 ảnh lỗi. Máy giúp khởi tạo vật cần tìm, nhưng chưa đạt chất lượng để Release. Nguồn: `qc/threshold_sweep.csv`, `qc/review_stats.json`, `qc/qc_report_auto.md`; số đo gold là in-sample, không suy rộng thành chất lượng batch.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Đúng class chưa đủ: trên 100% ảnh, kiểm riêng geometry theo phần nhìn thấy và điều kiện dùng ignore_region; không chấp nhận một box chỉ vì đúng lớp.
  2. Review như khi gán gold: trên từng ảnh, ẩn nhãn để quét tìm vật trước, rồi hiện nhãn và kiểm lần lượt bỏ sót → thừa → sai lớp → lỏng box; chỉ đánh dấu xong khi đủ bốn lượt.
  3. Không dùng confidence làm bằng chứng đúng: kiểm vị trí, kích thước và guideline cho mọi box/mask, kể cả common class điểm cao; mỗi ảnh có vật bị loại do điểm thấp phải kiểm lại vật đó trước khi kết thúc. Đây là quy tắc cho vòng rework, không phải khẳng định lượt đầu đã thực hiện đủ.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013.jpg | SAM 2.1 base (`sam2.1_b.pt`) | 3: 2 dương + 1 âm; thêm box | Đã xác nhận: loại người lái; giữ ghế/khung cố định của xe; không có hàng rời cần đưa vào mask. | Đã xác nhận biên, lỗ, kính và vùng sát người lái phù hợp guideline; xem `masks/part5_metadata.json`. | `img013_tuktuk_driver_excluded.jpg` |
| img_017.jpg | SAM 2.1 base (`sam2.1_b.pt`) | 3: 2 dương + 1 âm; thêm box | Đã xác nhận: không đưa người/hàng rời vào mask; giữ ghế và phần cố định của xe. | Đã xác nhận không tràn sang xe kế bên; vùng chạm, kính, bánh và lỗ phù hợp guideline; xem `masks/part5_metadata.json`. | `img017_foreground_tuktuk_separated.jpg` |
| img_004.jpg | SAM 2.1 base (`sam2.1_b.pt`) | 3: 2 dương + 1 âm; thêm box | Đã xác nhận: loại người đứng cạnh xe; không đưa người lái/hàng rời vào mask. | Đã xác nhận không tràn sang người/xe kế bên; kính chắn gió, khoảng hở cabin và gầm phù hợp guideline; xem `masks/part5_metadata.json`. | `img004_tuktuk_person_excluded.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? Đã xác nhận kiểm biên phần nhìn thấy ≤~3 px, lỗ nhỏ/lớn, kính/bóng và tách vật chạm nhau. Kính/cabin được làm rõ ở CAL-01; phạm vi giỏ/thùng theo lớp được chốt ở CAL-02. Ngưỡng nhìn thấy người thấp thoáng trong cabin còn chưa chốt; xem `guidelines/guideline_v1.1_calibrated.md`.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: Không dùng Interactor cho ba ví dụ này; chạy offline bằng `scripts/sam2_points.py`. Chưa có đánh giá sử dụng Interactor để đối chiếu.

## 6. Calibration (Part 6)

- Số quyết định: 4 đã xác nhận (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): CAL-02.
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: CAL-01 giữ kính, tách người và khoảng cabin mở; CAL-02 giỏ/thùng thuộc three_wheeler nhưng không mở rộng bicycle img_031 theo giỏ/thùng; CAL-03 kiểm lớp xe địa phương theo kết cấu; CAL-04 kiểm geometry độc lập trên mọi ảnh. Nguồn: `guidelines/guideline_v1.1_calibrated.md`.
- Nguyên nhân gốc: CAL-01/02: guideline mơ hồ; CAL-03: sai lớp dưới định nghĩa đã rõ; CAL-04: bỏ lọt hình học dưới quy tắc đã rõ. Nguồn: `qc/calibration_decisions.csv`.

## 7. QC và quyết định (Part 6)

- Mẫu: 5 (img_003, img_006, img_014, img_031, img_035; tổng 123 đối tượng theo `qc/qc_sample.csv`; học viên xác nhận phần cần kiểm người đã được review trong chat; vòng này đã biết kết quả bẫy, không phải kiểm mù hay người kiểm độc lập) ảnh (0 ảnh `can_review` + 5 ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: 14/123 = 11.4% (14 lỗi ghi trong phiếu, gồm 1 bỏ sót chưa có object_id; 123 đối tượng candidate trước sửa) · theo **ảnh**: 5/5 = 100.0%, theo `qc/qc_review.csv` và `qc/qc_report_auto.md` (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: sai vị trí/kích thước 7/14 lỗi (50.0%), ví dụ truck #2/#3 và person #6 ở img_003; thừa 6/14 (42.9%), ví dụ person giả #562 ở img_031; bỏ sót 1/14 (7.1%), motorbike gần x=662–724, y=459–527 ở img_035. Geometry lặp trên img_003/img_006/img_035, là lỗi hệ thống cần kiểm lại toàn batch, không chỉ sửa ảnh mẫu. Không có lỗi sai lớp/sai thuộc tính/sai mask được ghi trong phiếu này; không đồng nghĩa toàn batch không có các lỗi đó. Nguồn: `qc/qc_review.csv`.
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): với 5/5 ảnh lỗi, cận trên Wilson 95% là 100%. Tỉ lệ quan sát vượt ngưỡng 2%, nên không đủ điều kiện Release; chỉ 5 ảnh, tự kiểm và đã biết bẫy nên không ước lượng chính xác tỉ lệ lỗi thật của cả batch. Ngay cả một vòng 0/5 lỗi cũng có cận trên Wilson khoảng 43.4%, không chứng minh tỉ lệ lỗi thật ≤2%.
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): kiểm cấu trúc, taxonomy, đủ ảnh, box trong ảnh, sampling và tổng hợp phiếu; không xác nhận lớp/hình học đúng thực tế, vật cả người lẫn máy cùng bỏ sót hoặc việc thực hiện checklist. Nguồn: `docs/CALIBRATION_QC.md`.

**Quyết định: Hold**, ngày 2026-11-19. Lý do (đi theo Hold → Rework → Release): giữ quyết định của học viên để làm rõ và áp dụng nhất quán cách tách từng xe với ignore_region ở vùng xe đông/sát trước khi tái gán nhãn; câu hỏi #2 còn mở trong `guidelines/guideline_v1.1_calibrated.md`. Không dùng ignore_region để che các xe vẫn tách được ở img_035. Sau khi chốt vướng mắc này, chất lượng vẫn yêu cầu Rework: 5/5 ảnh lỗi >2%, geometry lặp 7/14 lỗi và bắt bẫy lượt đầu 6/12 <80%; không đủ điều kiện Release. `qc/qc_report_auto.md` gợi ý Rework với giả định không còn guideline gap, khác với Hold được giữ ở đây.
Nếu Rework / Hold, việc cần làm tiếp: Chốt ví dụ tách xe/ignore_region và bổ sung quyết định calibration; tái đánh nhãn theo guideline trên toàn 24 ảnh, ưu tiên các lỗi trong `qc/qc_review.csv` và 21 cảnh báo box nhỏ ở `qc/batch_validation.txt`. Export đầy đủ annotation của từng ảnh sửa vào `labels/qc_fixed.json`, rebuild candidate và chạy vòng QC mới có file bằng chứng riêng. Giữ nguyên nhãn/statistics lượt đầu và điểm bẫy 6/12; không thay bằng điểm rework đã biết bẫy. Hiện chưa có `labels/qc_fixed.json`, nên không coi sửa nhãn là đã hoàn tất.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): OpenCode (GPT-6.1 Sol) hỗ trợ setup GPU/CVAT, chạy dự đoán/merge/bẫy, export/validate/thống kê, tạo SAM mask và soạn calibration/báo cáo. Ba mask và CAL-01–04 đã được xác nhận trong chat. AI tạo phiếu QC cùng gợi ý Rework trong `qc/qc_review.csv` và `qc/qc_report_auto.md`; học viên xác nhận đã review phần cần kiểm người và giữ quyết định Hold. Giữ nguyên reviewer gốc trong phiếu để bảo toàn nguồn tạo; việc học viên review sau không biến vòng này thành kiểm mù hay độc lập. AI hỗ trợ tổng hợp số đo và phân tích từ log, chuẩn hóa ba quy tắc cá nhân thành thao tác kiểm được; không thay thế việc sửa nhãn. Kiểm tra bằng đối chiếu số đếm COCO, phiếu QC, confidence/IoU, hash template và các script kiểm submission.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Đọc guideline và tham khảo AI về cách hiểu về giai đoạn trước export ban đầu. Sau khi có kết quả dự đoán, đã yêu cầu đổi img_036 annotation #46 từ car sang ignore_region, giữ bbox; đây là ngoại lệ với quy tắc vật đơn lẻ <20 px. Bản gốc/audit ở `qc/gold_manual_before_revision.json` và `qc/gold_revision.json`.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows 11 Home Single Language 64-bit; RTX 3060 Laptop GPU 6 GB; CUDA 12.8, PyTorch 2.11.0+cu128; CVAT Docker v2.81.0, Project #2. Từng gặp Docker dừng và CVAT 502/đăng xuất; sau khởi động/đăng nhập đã tạo Task #11. Nguồn: `STUDENT.md`, `qc/batch_export.json`.
- Dùng cache `data/cache/` thay vì tự chạy? Không dùng dự đoán cache cho recipe chính; chạy YOLO26/YOLOE/SAM cục bộ trên CUDA. Nguồn: `model_log.csv`, `masks/part5_metadata.json`.
- Lỗi / bug script đã gặp (nếu có): không.
- Một điều bạn làm khác lần sau: chú ý hơn với các mô hình ML local, tránh thiên kiến tự động hóa.
