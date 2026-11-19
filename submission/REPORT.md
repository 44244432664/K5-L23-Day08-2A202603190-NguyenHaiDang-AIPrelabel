# Báo cáo Day 08 — <<MSHV - Họ và tên>>

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m — r01, r02; dự đoán thô r05 | person 0.15; motorbike 0.10; bicycle / car / bus / truck 0.30 (mặc định) | Học viên đã chấp nhận bộ ngưỡng này. Person và motorbike đạt mục tiêu recall trên gold; các lớp còn lại dùng mặc định do ít mẫu hoặc chưa đạt mục tiêu. So sánh bổ sung phát hiện pickup trong img_001 bị gọi truck, cho thấy vẫn phải duyệt lớp và hình học. |
| three_wheeler | YOLOE-26s — r03; dự đoán thô r06; tuk-tuk / cyclo / cargo tricycle → three_wheeler | 0.15 | Học viên chấp nhận ngưỡng do sweep đề xuất. Trên gold, ghép được 7/7 vật, nhưng có 2 proposal không khớp: precision 7/9, không phải kết quả hoàn hảo. YOLO26 không có lớp riêng cho xe ba bánh. |

### Các lần thử đã ghi nhận (Part 2)

Số liệu dưới đây lấy từ `model_log.csv` và đối chiếu `work/runs.csv`; nhận xét là phần tôi đã ghi trong log, chưa thay thế đánh giá bằng gold ở các Part sau.

| Run | Dữ liệu / thiết lập | Kết quả kỹ thuật | Thời gian chạy / ảnh |
|---|---|---|---|
| r01 | 8 ảnh pilot; YOLO26m; conf 0.25 | 215 box | 0.71 giây |
| r02 | 8 ảnh pilot; YOLO26m; conf 0.5 | 140 box | 0.52 giây |
| r03 | 8 ảnh pilot; YOLOE-26s; chỉ three_wheeler; conf 0.25 | 8 box | 0.27 giây |
| r04 | img_004.jpg, img_013.jpg; YOLO26m → SAM 2.1 base; conf 0.3; chỉ person / motorbike / car | 16 box có mask trên 2 ảnh | 4.05 giây |

- r04 thực tế chỉ giữ person / motorbike / car và cho 9 person, 6 car, 1 motorbike; không tính `truck` là kết quả của lượt chạy này.
- Các ví dụ gold và mask bổ sung bên dưới do trợ lý AI đối chiếu theo yêu cầu học viên; không được hiểu là học viên đã tự kiểm trực quan toàn bộ các ảnh. Những ví dụ gold không thay thế ghi chú quan sát trên 8 ảnh pilot của Part 2.
- Nguồn cho Part 3–4 đã chạy riêng trên CUDA ở conf 0.05, đủ 36 ảnh mỗi file: `work/prelabel/y26_all/pre.coco.json` (1672 box) và `work/prelabel/yoloe_all/pre.coco.json` (125 box). Đây là dự đoán thô, không phải nhãn đã duyệt hoặc số đo chất lượng.

- Lệnh `merge_prelabels.py` đã dùng: `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.15,motorbike=0.1,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`. Kết quả: 845 box trên 28 ảnh (24 batch + 4 gold); YOLOE được ưu tiên trước YOLO26.
- Cài bẫy: `python scripts/inject_traps.py --pre work/prelabel/merged_review/pre.coco.json --student "2A202603190"` tạo đủ 12 bẫy trên 12 ảnh batch. File import `prelabel/pre_traps_import.coco.json` có 24 ảnh / 741 box; file đo `prelabel/pre_traps.coco.json` có 28 ảnh / 845 box. Không mở `traps_key.json` trước khi hoàn tất review và export.
- **Phiên bản SAM đã dùng:** SAM 2.1 base (`sam2.1_b.pt`). Chọn làm phiên bản chính vì đây là lựa chọn cân bằng được repo khuyến nghị, checkpoint đã có sẵn, chạy CUDA thành công và không cần quyền gated như SAM 3. Ở r04: 16 mask / 2 ảnh, 4.05 giây/ảnh. Part 5 chạy thêm ba vật bằng click + box trên GPU. Chưa so sánh có kiểm soát với SAM phiên bản khác, nên không khẳng định đây là phiên bản tốt nhất.
- **Lỗi không sửa được chỉ bằng ngưỡng:** img_001, pickup được YOLO26 gọi `truck` ở conf 0.8757 trong khi label contract xếp bán tải vào `car`; img_012 có tuk-tuk bị gọi `car`. Tăng ngưỡng có thể bỏ proposal nhưng không đổi nó thành đúng lớp. Recipe dùng YOLOE cho three_wheeler, rồi người duyệt vẫn phải sửa lớp / bỏ trùng.
- **Điểm cao sai / điểm thấp khớp gold:** img_001, YOLO26 proposal #1 `truck` (0.8757) trùng pickup gold #1 `car`; img_036, proposal #1652 `motorbike` chỉ 0.1155 nhưng khớp gold #43 với IoU 0.964. Img_012, YOLOE proposal #56 `three_wheeler` 0.1939 khớp gold #31 với IoU 0.819. Confidence không phải xác suất đúng.
- **Quan sát mask bổ sung:** img_004, SAM tạo mask trên xe tuk-tuk màu xanh nhưng nhãn vẫn là `car` từ detector. Img_013, mask `car` lớn phủ cả người lái và không bao trọn phần sau xe; box/prompt và quy tắc tách người–xe vẫn cần kiểm. Có mask không có nghĩa lớp hoặc phần vật được chọn đã đúng.
- **Giới hạn và cách dùng:** chỉ 4 ảnh gold; chọn và đánh giá ngưỡng trên cùng gold gây lạc quan. Giữ ngưỡng mặc định cho lớp ít vật, không suy rộng các tỷ lệ này thành chất lượng của 24 ảnh batch, và vẫn review đủ mọi ảnh. Bản gold còn có sửa đổi sau khi đã thấy kết quả AI, được công bố ở mục 8.

### Kết quả Part 3 — ngưỡng đã được học viên chấp nhận

Nguồn: `qc/threshold_sweep.csv`, `.md`, `.png`. Script đã chạy với hai file dự đoán thô ở conf 0.05 và `labels/gold_manual.json`; ghép cùng lớp tại IoU ≥ 0.5, mục tiêu recall ≥ 90%. Chọn ngưỡng cao nhất trên lưới thử không quá 0.5 nếu lớp có ít nhất 5 vật gold và đạt mục tiêu.

| Lớp | Số vật gold | Ngưỡng hiệu lực đã chấp nhận | Precision = TP / (TP + FP) | Recall = TP / (TP + FN) | Cơ sở tự động |
|---|---:|---:|---|---|---|
| person | 24 | 0.15 | 22/37 = 59.5% | 22/24 = 91.7% | Đạt mục tiêu |
| bicycle | 4 | 0.30 (mặc định) | 2/5 = 40.0% | 2/4 = 50.0% | < 5 vật; không chọn riêng |
| motorbike | 11 | 0.10 | 10/20 = 50.0% | 10/11 = 90.9% | Đạt mục tiêu |
| car | 8 | 0.30 (mặc định) | 7/14 = 50.0% | 7/8 = 87.5% | Không ngưỡng thử nào ≤ 0.5 đạt mục tiêu |
| bus | 0 | 0.30 (mặc định) | 0/1 = 0.0% | Không xác định (0/0) | Không có vật gold; không chọn riêng |
| truck | 1 | 0.30 (mặc định) | 0/2 = 0.0% | 0/1 = 0.0% | < 5 vật; không chọn riêng |
| three_wheeler | 7 | 0.15 | 7/9 = 77.8% | 7/7 = 100.0% | Đạt mục tiêu trên mẫu nhỏ |

Chuỗi ngưỡng học viên đã chấp nhận để dùng ở Part 4:

```text
--thr "person=0.15,motorbike=0.1,three_wheeler=0.15" --conf 0.3
```

Các lớp không liệt kê trong `--thr` dùng `--conf 0.3`. Đó là giá trị dự phòng, **không có nghĩa** các lớp này đã đạt recall 90%. Merge cho Part 4 đã chạy; các số trong bảng này vẫn là kết quả quét dự đoán thô, không phải chất lượng nhãn sau review.

- **Xác nhận:** học viên yêu cầu chấp nhận các ngưỡng đề xuất trong hội thoại; sweep chạy lại sau thay đổi gold vẫn cho cùng chuỗi. Không chọn ngưỡng riêng cho bicycle (4 vật), bus (0), truck (1). Recall bus không xác định do mẫu số 0, dù CSV dùng giá trị quy ước 0.0.
- **Đánh đổi được ghi nhận:** giữ recall person 22/24 và motorbike 10/11 nhưng precision chỉ 22/37 và 10/20, tức còn nhiều proposal phải kiểm. Không tăng ngưỡng chỉ để bảng trông sạch hơn; lỗi sai lớp và proposal trùng vẫn phải xử lý bằng review. Car không đạt mục tiêu 90% và không được làm tròn thành đạt.

### Đối chiếu ảnh gốc, gold và dự đoán ở ngưỡng đã chấp nhận

Nguồn số liệu chi tiết: `qc/threshold_comparison.json`. Trợ lý AI đã đọc bốn ảnh gốc và các overlay gold / YOLO26 / YOLOE theo yêu cầu; ảnh so sánh local nằm ở `work/part3_comparison/img_0xx_comparison.jpg`. Đây là hai nguồn dự đoán thô **trước khi merge**, loại proposal nằm trong ignore_region khi tính số, ghép cùng lớp với IoU ≥ 0.5.

| Ảnh | YOLO26 TP / FP / FN | YOLOE TP / FP / FN | Điểm đối chiếu |
|---|---|---|---|
| img_001.jpg | 18 / 18 / 2 | 0 / 0 / 0 | Pickup tiền cảnh bị gọi truck ở điểm cao; gold car #1 không được ghép cùng lớp. |
| img_012.jpg | 6 / 7 / 0 | 4 / 2 / 0 | YOLO26 gọi tuk-tuk là car; YOLOE ghép đủ 4 three_wheeler nhưng vẫn có 2 proposal không khớp gold. |
| img_020.jpg | 6 / 6 / 1 | 3 / 0 / 0 | Vật gold #40 truck ở mép phải được gọi bus (0.6908); cần đối chiếu định nghĩa bus/truck thay vì sửa bằng threshold. |
| img_036.jpg | 11 / 7 / 4 | 0 / 0 / 0 | Còn sai khác person/bicycle với gold; có proposal motorbike điểm thấp nhưng IoU cao và các proposal lồng nhau cần kiểm. |

YOLO26 chỉ được so với sáu lớp COCO, YOLOE chỉ so với three_wheeler; 0/0/0 của YOLOE trên img_001/img_036 nghĩa là không có gold hay proposal của lớp này, không phải phép đo “chính xác 100%”. FP/FN ở đây là sai khác với gold, không tự chứng minh mọi proposal không ghép được đều là vật giả; hình học, ignore_region và chính gold cũng cần kiểm.

**Sửa đổi tham chiếu đã công bố:** theo yêu cầu học viên, annotation #46 trong img_036 đổi từ car sang ignore_region, giữ nguyên bbox. Bản gốc ở `qc/gold_manual_before_revision.json`, audit ở `qc/gold_revision.json`. Đây là ngoại lệ do học viên yêu cầu: v1.0 thường bỏ nhãn một vật đơn lẻ <20 px và chỉ dùng ignore_region cho nhóm không phân biệt được. Sau sửa, car từ 9 xuống 8 vật, recall từ 8/9 xuống 7/8; không trình bày thay đổi này như một cải thiện chất lượng. Chưa đồng bộ thay đổi vào CVAT Task #10 vì phiên trình duyệt đã đăng xuất.

## 2. AI error profile (từ `qc/review_stats.md`)

Học viên xác nhận đã review 24 ảnh trên Task `batch-review` #11 / Job #11 và báo thời gian active đã ghi là **74 phút**. Trợ lý lưu nhãn trong CVAT rồi export COCO 1.0 không kèm ảnh thành `labels/batch_reviewed.json`: **24 ảnh / 655 annotation**, gồm 651 box đối tượng và 4 ignore_region, so với baseline import 741 box. Export được lưu **trước khi trợ lý đọc key để chấm**; không thể xác minh độc lập việc học viên đã tuân thủ từng bước checklist.

`validate_labels.py --all` không có lỗi cấu trúc, nhưng còn **21 cảnh báo box cao <20 px trên 9 ảnh**; không tự động xoá/đổi nhãn các box này. Chi tiết: `qc/batch_validation.txt` và `qc/batch_export.json`. Snapshot lượt đầu: `qc/batch_reviewed_first_pass.json`, `qc/review_stats_first_pass.{json,md}`; ZIP export gốc được lưu cục bộ tại `work/part4_export/batch_reviewed_original_cvat.zip`.

Đã chạy `review_stats.py` với **full baseline** `prelabel/pre_traps.coco.json`, export đã review, `labels/gold_manual.json` và key. Bảng dưới loại vùng bẫy/ignore theo script; mẫu số không phải 741 hay 655. Có 28 đối tượng tổng cộng ở hai phía pre/final bị loại quanh vùng bẫy; 13 box máy không ghép được nhưng nằm trong ignore_region cuối không tính deleted, nên accepted + edited + deleted không cộng đủ 725.

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 591 | 81.5% | 725 box máy |
| edited | 17 | 2.3% | 725 box máy |
| deleted | 104 | 14.3% | 725 box máy |
| added | 31 | 4.9% | 639 đối tượng sau duyệt |

- Trong deleted + added có **11 trường hợp đổi lớp**; số sót thật theo ước lượng của script là 31 − 11 = **20**, không phải một phép đo recall độc lập trên batch.
- **Lớp cần kiểm nhiều nhất theo lượt review này:** truck có 5 deleted / 6 box máy được tính (83.3%), nhưng mẫu rất nhỏ; three_wheeler có 19 deleted / 38 (50.0%), lớn hơn về số lượng. Ví dụ img_013 có 2 truck không ghép được và 2 car cuối không ghép được; img_003 có 5 three_wheeler không ghép được, còn img_014 có 3. Đây là bằng chứng thay đổi giữa pre/final, không tự chứng minh mọi thao tác của người review đều đúng hay nguyên nhân là model; còn cần QC ảnh.
- So với gold: precision của **máy** = **48/(48+34) = 58.5%** (KTC 95% khoảng 48–69%); recall = **48/(48+7) = 87.3%** (KTC khoảng 76–94%), trên **55 đối tượng / 4 ảnh gold**. Ngưỡng được chọn trên chính gold này, nên kết quả in-sample lạc quan; gold còn có sửa đổi hậu kiểm đã công bố ở mục 1. Batch export không có gold, vì vậy **không đo residual/new error sau review**.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Lượt đầu bắt 6/12 = 50.0%**, KTC 95% khoảng 25–75%; **không đạt ≥80% (cần ít nhất 10/12)**. Theo loại: extra 2/3, wrong_class 1/3, missing 2/3, loose 1/3.
- Bẫy lọt: **img_008 (extra), img_003 (wrong_class), img_005 (missing), img_010 (wrong_class), img_031 (loose), img_006 (loose)**.
- Vì sao lọt / bước checklist thực sự đã bỏ: **<<học viên tự xác nhận; không suy diễn từ số đo>>**. Bẫy so với box gốc của máy, nên nếu sửa hợp lý nhưng khác box gốc cần đối chiếu ảnh và ghi chú, không tự coi mọi ca chấm trượt là lỗi người review.
- Accepted **81.5%** đi cùng bắt bẫy **50%** cho thấy tỷ lệ giữ nhãn cao không chứng minh đã phát hiện đủ lỗi; sai lớp và hình học là hai loại bắt thấp nhất. Kết quả tương thích với nguy cơ thiên kiến tự động hóa nhưng không chứng minh động cơ hay quy trình thực tế của học viên.
- **Trạng thái Part 4: Rework về chất lượng.** Export, validation và thống kê lượt đầu đã xong; cần duyệt lại toàn bộ 24 ảnh, ưu tiên các ảnh bẫy lọt và 21 cảnh báo <20 px ở img_014, 017, 018, 023, 024, 025, 027, 033, 034. Không biến vật đơn lẻ quá nhỏ thành ignore_region nếu guideline chỉ cho vùng nhóm không phân biệt được.
- Giữ nguyên baseline/key và snapshot lượt đầu. Sau rework export thành file riêng và chấm với `--out` riêng; vì key đã được dùng/chia sẻ kết quả, đó là **lượt sửa có biết đáp án**, không được thay thế điểm mù ban đầu bằng điểm mới.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): duyệt **74/24 = 3.08 phút/ảnh**, học viên báo là thời gian active đã ghi. **Chưa có thời gian gold_manual**, nên chưa tính được tiết kiệm thời gian hay so sánh gán tay/duyệt. Không lấy thời gian tạo/cập nhật task hoặc thời gian chạy script thay thời gian làm việc.
- **Kết luận tạm theo số đo:** pre-label hữu ích như bản nháp, chưa đủ bằng chứng khẳng định tiết kiệm thời gian hay chấp nhận chất lượng sau review. Person giữ được 338/378 = 89.4% và motorbike 181/227 = 79.7% trong profile, nhưng thiếu đối chiếu batch độc lập; truck/three_wheeler cần kiểm kỹ và bắt bẫy 50% buộc rework. Bốn ảnh gold đã làm trước giúp quen cảnh, độ khó 4 gold/24 batch khác nhau, nên kể cả có đủ thời gian thì so sánh vẫn có thiên lệch.
- **Ba biện pháp đề xuất từ kết quả, học viên cần xác nhận áp dụng** (không khẳng định đã làm ở lượt đầu):
   1. Ẩn nhãn và quét/đếm vật ít nhất ~15 giây ở **24/24 ảnh**, ảnh đông quét theo góc phần tư, rồi mới hiện nhãn.
   2. Mỗi ảnh hoàn tất đủ **4 lượt kiểm missing → extra → wrong_class → loose**; ghi nhận hoàn thành từng lượt, kiểm toàn bộ truck/three_wheeler và box <20 px theo guideline.
   3. Nghỉ 3–5 phút sau mỗi ~10 ảnh, ghi active time riêng với nghỉ; giữ export trước khi đọc key, chỉ coi đạt target khi ≥10/12 và không dùng điểm sau biết key để tuyên bố lượt đầu đạt.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013.jpg — tuk-tuk tiền cảnh lớn | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 dương, 1 âm; thêm box) | AI quan sát, học viên xác nhận: người lái tại click âm được loại; không có hàng rời cần đưa vào mask; ghế/khung cố định thuộc xe. | 274,333 px = 39.3% ảnh; 1 polygon. Học viên xác nhận thân/mái, biên, lỗ, kính và vùng sát người lái phù hợp guideline. | `img013_tuktuk_driver_excluded.jpg` |
| img_017.jpg — tuk-tuk tiền cảnh chạm/che xe kế bên | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 dương, 1 âm; thêm box) | AI quan sát, học viên xác nhận: không có người lái/hàng rời cần đưa vào mask; ghế và phần cố định của xe được phủ. | 88,672 px = 12.7% ảnh; 1 polygon. Học viên xác nhận vùng chạm, kính/cửa sổ, bánh, lỗ và việc không tràn sang xe kế bên đạt yêu cầu. | `img017_foreground_tuktuk_separated.jpg` |
| img_004.jpg — tuk-tuk phải cạnh người đứng và các xe khác | SAM 2.1 base (`sam2.1_b.pt`) | 3 (2 dương, 1 âm; thêm box) | AI quan sát, học viên xác nhận: người đứng tại click âm bị loại; không có người lái/hàng rời cần đưa vào mask. | 92,719 px = 13.3% ảnh; 1 polygon. Học viên xác nhận kính chắn gió, khoảng hở cabin, chi tiết nhô/gầm và việc không tràn sang người/xe kế bên đạt yêu cầu. | `img004_tuktuk_person_excluded.jpg` |

- **Đối chiếu guideline 4.2 (AI chuẩn bị, học viên xác nhận ngày 2026-11-19):** quy tắc hữu ích gồm bám biên phần nhìn thấy (sai số mục tiêu ≤~3 px), bỏ lỗ lớn nhưng cho phép lấp lỗ <5%, loại bóng/giữ kính, và tách hai vật chạm nhau không chồng polygon. Quy tắc lớp ở mục 2/4.1 cũng buộc tách người lái khỏi xe. Guideline còn thiếu quyết định dứt khoát cho hàng nhô ra, kính nhìn xuyên thấy nội thất/người, và ngưỡng người thấp thoáng trong cabin; đó vẫn là câu hỏi mở mục 6, không để SAM tự quyết.
- Dữ liệu kỹ thuật, prompt, bbox, hash, nhận xét AI và phạm vi xác nhận của học viên: `masks/part5_metadata.json`. Tờ so sánh ảnh gốc/mask cục bộ: `work/part5_review/part5_mask_review_sheet.jpg`. Các overlay được tạo tự động; học viên đã xác nhận bảy kiểm tra hình học/nội dung/lớp được liệt kê trong chat.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: không dùng cho ba ví dụ này; chạy offline bằng `sam2_points.py`. Offline đáp ứng phần cốt lõi nhưng không ghi polygon trực tiếp vào CVAT, còn Interactor cho phép lặp click ngay trên canvas.

## 6. Calibration (Part 6)

- `calibration_decisions.csv` có **4 quyết định có bằng chứng** và đúng nguồn yêu cầu: 2 `cau_hoi_mo`, 1 `loi_may`, 1 `bay_lot`. CAL-01 được học viên xác nhận qua Part 5; học viên xác nhận CAL-02–04 trong chat ngày 2026-11-19.
- **CAL-01 (đã xác nhận):** kính thuộc mask xe, nhưng người/nội thất nhìn xuyên qua không trở thành một phần xe; khoảng cabin mở lớn không lấp như kính. Ví dụ: img_004, img_013, img_017.
- **CAL-02 (đã xác nhận):** giỏ/thùng/container thuộc `three_wheeler`; với bicycle img_031, box/mask không mở rộng theo giỏ/thùng/container. Bbox chữ nhật vẫn có thể chồng lên cargo nằm bên trong phạm vi khung/bánh.
- **CAL-03 (đã xác nhận):** nhầm three_wheeler thành car/motorbike/bicycle/truck là lỗi máy/người dưới quy tắc v1.0 đã rõ; thêm lượt kiểm lớp xe địa phương, không đổi semantic lớp.
- **CAL-04 (đã xác nhận):** hai bẫy loose ở img_006/img_031 lọt dù quy tắc box đã rõ; thêm lượt geometry độc lập, không coi đây là lỗ hổng định nghĩa vật.
- `guidelines/guideline_v1.1_calibrated.md` là supplement v1.1, có ảnh/case/history. Các câu mở #1, #2, #4, #6, #7 chưa có bằng chứng đủ rõ để chốt nhưng không phát sinh blocker trong 5 ảnh QC đã chấm; nếu gặp ca tương ứng phải tick `can_review`.
- Phân loại nguyên nhân đã xác nhận: CAL-01/02 là guideline còn mơ hồ; CAL-03 là lỗi máy/lệ thuộc gợi ý; CAL-04 là không bắt lỗi hình học dưới quy tắc đã rõ. Chưa có `qc_fixed.json`; các lỗi QC vừa tìm thấy cần được sửa trong vòng Rework.

## 7. QC và quyết định (Part 6)

- Candidate đã tạo từ `batch_reviewed.json`, chưa có `qc_fixed`: **24 ảnh / 655 annotation**, không có ảnh được thay thế; xem `release/release_manifest.json`.
- Mẫu seed 42: **5 ảnh** (**0** ảnh `can_review` + **5** ngẫu nhiên), **123 đối tượng**: img_003 (27), img_006 (35), img_014 (22), img_031 (4), img_035 (35). Không có box nào tick `can_review`, nên không có ca khó bổ sung. Theo yêu cầu học viên, OpenCode AI đối chiếu ảnh gốc với overlay ID và điền phiếu; đây **không phải QC người độc lập**. Ba ảnh bẫy lọt img_003/img_006/img_031 cùng img_035 có bẫy đã bắt tình cờ nằm trong mẫu, nên vòng này có thông tin, không phải self-check mù.
- **Lỗi theo đối tượng:** **14/123 = 11.4%**. **Ảnh có lỗi:** **5/5 = 100.0%**, vượt ngưỡng Release ≤2%.
- Top lỗi: `sai_vi_tri_kich_thuoc` **7/14** (img_003, img_006, img_035), `thua` **6/14** (img_006, img_014, img_031, img_035), `bo_sot` **1/14** (img_035). Geometry chiếm đúng 50% và lặp ≥3 lần, nên script coi là lỗi hệ thống.
- Điểm sửa chính: truck #2/#3 và person #6 ở img_003; duplicate/box gộp motorbike #82/#84/#85 ở img_006; person #166 cao 19.3 px ở img_014; person giả #562 trên quần áo trưng bày ở img_031; ignore #633/#634, motorbike bị sót, three_wheeler bị tách #651/#652 và person #635 ở img_035. Bicycle #561 không bị tính lỗi: bbox theo outer extent của khung/bánh, còn cargo nằm bên trong hình chữ nhật là overlap không tránh được.
- **Giới hạn mẫu nhỏ:** với 5/5 ảnh lỗi, cận trên Wilson 95% là **100%**. Mẫu nhỏ và QC do AI hỗ trợ không ước lượng chính xác tỉ lệ lỗi thật; tuy nhiên 5/5 đã đủ bác bỏ điều kiện quan sát ≤2%, không thể dùng để Release.
- Script kiểm được cấu trúc, phạm vi box, đủ ảnh, số đếm, sampling và tổng hợp phiếu; không biết box ôm vật thật, lớp đúng thực tế, vật cả người lẫn máy cùng bỏ sót, hoặc checklist đã được tuân thủ.

**Quyết định: Rework, ngày 2026-11-19.** Không có guideline/data blocker quan sát được sau khi học viên chốt CAL-02–04, nên không chọn Hold. Chọn Rework vì 5/5 ảnh QC có lỗi >2%, geometry là lỗi hệ thống 7/14, và bắt bẫy lượt đầu 6/12 = 50% <80%. Không đủ điều kiện Release.

Việc tiếp theo: sửa các lỗi ghi trong `qc_review.csv` trên CVAT; export **đầy đủ mọi annotation** của từng ảnh đã sửa vào `labels/qc_fixed.json`; giữ `qc_review_pre_ai.csv` / `qc_report_auto_pre_ai.md` và vòng QC hiện tại; rebuild candidate, tạo vòng mẫu mới có bằng chứng riêng, rồi mới đánh giá lại Hold → Rework → Release.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** OpenCode (GPT-6.1 Sol) hỗ trợ cài đặt GPU/CVAT, chạy script, chuẩn hóa câu chữ từ ghi chú của học viên; theo yêu cầu học viên, trợ lý còn đối chiếu ảnh gold với dự đoán, mô tả ca lỗi/mask và cập nhật phần kỹ thuật mục 1. Các ngưỡng được học viên chấp nhận trong hội thoại. Part 4: trợ lý merge/cài bẫy, tạo và import Task #11, sau khi học viên báo review xong thì lưu/export, validate, chấm và điền mục 2–4 từ số đo; học viên báo 74 phút active. Trợ lý không tự sửa nhãn batch sau khi chấm; nguyên nhân bẫy lọt và ba biện pháp đề xuất chưa được học viên xác nhận. Part 5: trợ lý tạo ba mask/nhận xét ban đầu, học viên xác nhận bảy kiểm tra trong chat. Part 6: trợ lý build candidate/lấy mẫu/soạn calibration; học viên chốt CAL-02–04, rồi yêu cầu trợ lý tự động đối chiếu ảnh gốc–overlay, điền 14 lỗi QC và áp cây quyết định. Kết quả Rework là AI-assisted, không phải QC người độc lập. Kiểm tra đã thực hiện: đối chiếu log, số đếm COCO, ghép IoU và validate_labels. Học viên tự xác nhận lại các nhận xét AI trước khi nộp: <<xác nhận của học viên>>
- Gold có bị nhiễm không: trước export ban đầu, học viên tự xác nhận việc không xem AI: <<chưa xác nhận>>. Sau khi đã có dự đoán / số đo AI, học viên yêu cầu đổi img_036 annotation #46 car → ignore_region; trợ lý thực hiện đúng một đổi lớp, lưu gold gốc và audit trong `qc/`, rồi chạy lại sweep. Gold hiện tại có sửa đổi hậu kiểm sau AI, không được mô tả là tham chiếu hoàn toàn mù / độc lập; xem mục 1 và `qc/gold_revision.json`.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
