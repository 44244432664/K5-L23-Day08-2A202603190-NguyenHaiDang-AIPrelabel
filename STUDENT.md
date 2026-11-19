# Thông tin học viên

- Họ và tên: <<>>
- MSHV: 2A202603190
- Lớp (L2 / L3 / khác): <<>>
- Tài khoản GitHub: 44244432664
- Repo bài nộp (URL): https://github.com/44244432664/K5-L23-Day08-2A202603190-NguyenHaiDang-AIPrelabel
- Commit chốt bài (hash): <<>>
- Máy: Windows 11 Home Single Language 64-bit; NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM), CUDA 12.8 (PyTorch 2.11.0+cu128)
- CVAT (Docker): v2.81.0; Project `day08-2A202603190` — http://localhost:8080/projects/2

<!-- MSHV và tài khoản/URL GitHub lấy từ git remote origin; máy/GPU từ kiểm tra hệ thống và CUDA; CVAT từ image Docker và Project đã tạo, kiểm tra trong giao diện. Họ tên đầy đủ có dấu, lớp, commit chốt và kết quả chưa được xác nhận nên giữ nguyên chỗ điền. -->

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 591 / 17 / 104 (mẫu số: 725 box máy sau loại vùng bẫy); added: 31 (mẫu số: 639 đối tượng sau duyệt). Có 13 box máy không ghép được nhưng nằm trong ignore_region nên không tính deleted. Nguồn: `submission/qc/review_stats.json`.
- So với gold: precision / recall của **máy** 48/82 = 58.5% / 48/55 = 87.3% (4 ảnh gold; in-sample).
- Bắt bẫy: 6/12 = 50.0% ở lượt đầu; dưới ngưỡng 80% (cần ≥10/12), phải rework; không sửa nhãn sau khi chấm để nâng điểm lượt đầu.
- Thời gian: gán tay <<... phút/ảnh>>; duyệt có pre-label 74/24 = 3.08 phút/ảnh (74 phút do học viên báo là thời gian active đã ghi).
- Lỗi QC AI-assisted: 100.0% theo ảnh (5/5), 11.4% theo đối tượng (14/123); không phải QC người độc lập.
- **Quyết định:** Rework — 5/5 ảnh mẫu có lỗi, geometry lặp 7/14 lỗi và bắt bẫy lượt đầu chỉ 6/12.
