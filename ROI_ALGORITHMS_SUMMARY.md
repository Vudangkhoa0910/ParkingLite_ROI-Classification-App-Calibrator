# ROI Algorithm Summary (Ý chính)

## 1) ROI là gì trong project này
- ROI = vùng quan tâm (mỗi ô đỗ xe là 1 ROI).
- App không phân tích toàn bộ ảnh, chỉ cắt ROI của từng slot để phân loại OCC/FREE.

## 2) Luồng xử lý chung
1. Ảnh reference (không xe) + ảnh test (có thể có xe).
2. Warp perspective ROI về kích thước cố định 32x32.
3. Tiền xử lý ROI: CLAHE + Gaussian blur.
4. Tính metric: SSIM, diff ratio, edge ratio, MAD, histogram, variance, block stats.
5. Chạy thuật toán được chọn -> trả về:
   - occupied (bool)
   - confidence (0..1)
   - metric_name + metric_value

## 3) Các thuật toán đang có trong app

### legacy_ssim_edge
- Rule cũ:
  - occupied khi `(ssim < nguong && diff_ratio > nguong) OR edge_ratio > nguong`.
- Nhanh, đơn giản, nhưng dễ false-positive khi edge nhiều.

### ssim_diff_gate
- Bỏ nhanh edge OR.
- Chỉ dùng gate SSIM + diff_ratio.
- Thường ổn định hơn legacy trong bộ dữ liệu hiện tại.

### mad
- Mean Absolute Difference giữa ROI test và ROI ref.
- metric cao -> khả năng có xe cao.

### gaussian_mad
- MAD có trọng số Gaussian (trung tâm ROI được ưu tiên hơn).
- Giảm ảnh hưởng lệch mép ROI.

### block_mad
- Chia ROI thành block 8x8, block nào vượt ngưỡng thì "vote".
- Có xe khi tỷ lệ vote vượt ngưỡng.
- Hợp với trường hợp thay đổi cục bộ.

### percentile_mad
- Dùng P75 của |diff| thay vì mean.
- Mạnh khi xe chỉ che một phần ROI.

### max_block_mad
- Lấy block thay đổi mạnh nhất.
- Dễ bắt xe che một góc ROI.

### histogram_intersection
- So histogram xám 16 bins giữa ref/test.
- intersection thấp -> khác biệt cao -> có xe.

### variance_ratio
- So tỷ lệ variance(test)/variance(ref).
- Xe thường làm tăng texture/variance.

### hybrid_mad_hist
- Kết hợp MAD + percentile + max_block + histogram bằng weighted score.
- Mục tiêu: giảm lỗi đơn feature.

### roi_ensemble (recommended)
- Tổng hợp nhiều metric bằng weighted score.
- Có confidence và độ bền cao hơn trong pipeline hiện tại.

## 4) Cách chọn nhanh trong thực tế
- Muốn baseline đơn giản: `mad` hoặc `ssim_diff_gate`.
- Muốn robust hơn: `roi_ensemble`.
- Muốn debug từng góc nhìn: test lần lượt `block_mad`, `max_block_mad`, `histogram_intersection`, `variance_ratio`.

## 5) Ghi chú
- Các threshold đang tune cho pipeline GUI hiện tại (warp ROI + CLAHE + blur).
- Nếu đổi camera/góc đặt camera/ánh sáng, nên benchmark lại và tune ngưỡng.