# Hướng Dẫn Sử Dụng Tools (ROI Calibrator)

Tài liệu này hướng dẫn cách sử dụng nhanh các tools/mode trong ứng dụng ROI Calibrator.

## 1. Tổng quan các mode

- Draw Slot: Vẽ từng slot bằng 4 điểm.
- Edit / Move: Chọn và chỉnh sửa slot (kéo góc, kéo cả slot, kéo cạnh).
- Select Grid: Chọn và chỉnh sửa cả nhóm grid (group-level).
- Auto Grid: Vẽ 4 góc khu vực và tạo grid tự động theo Rows/Cols.
- Tile Box: Vẽ 1 template 4 góc, chia theo Tile Rows/Tile Cols, kéo để nhân theo cụm.

## 2. Quy trình sử dụng nhanh

1. Bấm Open Image để mở ảnh.
2. Chọn mode cần dùng trên thanh công cụ.
3. Vẽ/chỉnh ROI trên canvas.
4. Bấm Save Config để xuất JSON/C/Python.

## 3. Auto Grid

1. Chuyển sang Auto Grid (hoặc phím G).
2. Ở MODE SETTINGS, nhập Auto Grid Rows và Cols.
3. Click 4 điểm theo thứ tự TL -> TR -> BR -> BL.
4. App tạo 1 grid group.
5. Chuyển sang Select Grid/Edit để chỉnh lại cả nhóm hoặc từng slot.

## 4. Tile Box

1. Chuyển sang Tile Box (hoặc phím T).
2. Ở MODE SETTINGS, nhập Tile Rows và Tile Cols (tách riêng với Auto Grid).
3. Click 4 điểm tạo template box.
4. Click-giữ bên trong template và kéo để nhân theo cụm.
5. Thả chuột để áp dụng cụm tile.
6. Sau khi áp dụng, có thể chỉnh theo group giống auto grid thông qua handle grid.

Ghi chú quan trọng:
- Tile Rows/Cols chỉ dùng cho Tile Box.
- Auto Grid Rows/Cols chỉ dùng cho Auto Grid.
- Hai bộ thông số này độc lập nhau.

## 5. Chỉnh sửa (Edit/Select)

- Edit / Move:
  - Kéo góc để chỉnh hình dạng 1 slot.
  - Kéo thân slot để di chuyển slot.
  - Kéo cạnh để tạo slot kề bên.
- Select Grid:
  - Chọn nhóm grid.
  - Kéo corner/edge để biến đổi cả nhóm.

## 6. Actions hiện có

- Auto Number (N): Đánh số lại slot.
- Delete Selected (Del): Xóa slot hoặc xóa cả grid group đang chọn.
- Duplicate (D): Nhân bản slot đang chọn.
- Clear All: Xóa tất cả slot.
- Undo / Redo: Hoàn tác / Làm lại.

## 7. Phím tắt (chỉ dùng phím chữ)

- D: Draw Slot
- E: Edit / Move
- G: Auto Grid
- Q: Select Grid
- T: Tile Box
- N: Auto Number
- S: Save Config
- R: Run ROI Classification
- U: Undo
- Ctrl+Z: Undo
- Ctrl+Y hoặc Ctrl+Shift+Z: Redo
- Delete / BackSpace: Delete Selected
- Ctrl+D: Duplicate
- Esc: Cancel thao tác đang vẽ
- + / - / F: Zoom in / Zoom out / Fit


## 8. Lưu file JSON và lịch sử theo từng ảnh

- Mỗi ảnh có 1 file auto JSON riêng: <ten_anh>.roi.json.
- Đường dẫn file auto JSON nằm cùng thư mục với ảnh.
- Lịch sử Undo/Redo được lưu riêng theo từng file JSON của từng ảnh.
- Khi đóng app, trạng thái hiện tại và history được lưu lại.

## 9. Mẹo thao tác để tránh lỗi

- Vẽ 4 điểm theo vòng liên tục (không cắt chéo) để được shape ổn định.
- Nếu đang thao tác Tile mà muốn dừng nhanh, dùng Esc hoặc đổi mode trực tiếp.
- Nếu không thấy mode settings, hãy chuyển đúng mode (Grid hoặc Tile) để panel hiện đúng thông số.

