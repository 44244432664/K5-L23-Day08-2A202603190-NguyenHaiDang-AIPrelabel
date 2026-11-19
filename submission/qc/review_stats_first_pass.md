# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 725; đối tượng sau duyệt: 639; ngưỡng accepted: IoU >= 0.9
- Đã **loại 28 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 591 | 591/725 = 81.5% | 725 box máy |
| edited (sửa) | 17 | 17/725 = 2.3% | 725 box máy |
| deleted (xoá) | 104 | 104/725 = 14.3% | 725 box máy |
| added (vẽ thêm) | 31 | 31/639 = 4.9% | 639 đối tượng sau duyệt |

Trong số deleted + added, **11 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 20.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 338 | 3 | 37 | 5 |
| bicycle | 5 | 1 | 1 | 3 |
| motorbike | 181 | 12 | 34 | 10 |
| car | 46 | 0 | 7 | 6 |
| bus | 2 | 0 | 1 | 0 |
| truck | 1 | 0 | 5 | 3 |
| three_wheeler | 18 | 1 | 19 | 4 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 55 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 48 | 34 | 7 | 58.5% (KTC 95%: 48–69%) | 87.3% (KTC 95%: 76–94%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 6/12 = 50.0% (KTC 95%: 25–75%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 2/3
  - wrong_class: 1/3
  - missing: 2/3
  - loose: 1/3
  - lọt: img_008.jpg (extra), img_003.jpg (wrong_class), img_005.jpg (missing), img_010.jpg (wrong_class), img_031.jpg (loose), img_006.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
