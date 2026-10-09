# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

## Thông tin học viên

- Họ tên: Lê Nguyễn Trâm Anh
- MSSV: 2A202602760
- Email: kathyforwork05@gmail.com
- Link repo (fork): https://github.com/itskathy05/K4-L3-DAY23-LeNguyenTramAnh-2A202602760-SensorFusion

## Kết quả official

- Chế độ: `compare`; frames `0–198` (199 frames); seed `0`.
- Segment: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`.
- Detection: precision `0.97009`, recall `0.70040`, TP/FP/FN `519/16/222`.
- LiDAR tracking: RMSE `0.15032 m`, matches `502`, sum squared error `11.343649 m²`, ghost track-frames `0`, missed GT `239`, mean confirmed tracks `2.52261`.
- Fused tracking: RMSE `0.13587 m`, matches `502`, sum squared error `9.266812 m²`, ghost track-frames `0`, missed GT `239`, mean confirmed tracks `2.52261`.
- Tracking precision: `1.0` cho cả hai mode. Coverage: `502/519 = 0.96724` cho cả hai mode.

Hai mode có cùng 502 matches, 0 ghost và 239 missed GT; detection và coverage không đổi. Camera giảm sum squared error `18.3%` và RMSE khoảng `0.01446 m` (`0.15032 → 0.13587 m`), tức cải thiện độ chính xác vị trí của các track đã ghép.

Lệnh chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

RMSE là căn bậc hai của `sum_sq_err/matches`, tính trên vị trí 3D của confirmed tracks ghép một-một với GT trong cửa sổ BEV, với gate XY 2.0 m. Camera trong lab dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded làm measurement.

`student/artifacts/grade_run.log` có đúng một record cho mỗi cặp `(mode, frame)`. Các invariant `matches+ghosts==confirmed` và `matches+misses==valid_gt` thỏa mãn; metrics tái tạo từ log khớp `metrics.json`. Sáu artifacts gồm `metrics.json`, `metrics_lidar.json`, `metrics_fused.json`, `grade_run.log`, `grade_run_lidar.log`, `grade_run_fused.log`.

## Giải thích Parts E–H

1. **Đo LiDAR và camera trong EKF.** LiDAR measurement `z` là vị trí 3D `[x,y,z]` với covariance `R` kích thước 3×3. Camera measurement `z` là pixel `[u,v]` với covariance 2×2 lấy từ `sigma_cam_i²` và `sigma_cam_j²`. Mỗi sensor có mô hình quan sát và Jacobian riêng; innovation covariance là `S=HPHᵀ+R`. Xem `student/workspace/kalman.py`, `camera_fusion.py` và `track_management.py`.

2. **Mahalanobis gating.** Squared Mahalanobis chuẩn hóa residual bằng innovation covariance nên xét đến uncertainty và khác biệt đơn vị đo. So sánh với phân vị χ² theo số chiều measurement (`gating_threshold`) để loại cặp không hợp lý trước greedy assignment, tránh measurement ngoại lai cập nhật sai track. Kiểm tra FOV được thực hiện trước projection và distance.

3. **Track-then-fuse.** Runtime dùng một tracker chung: predict mỗi frame, association/update LiDAR (`AssocL`), sau đó association/update camera (`AssocC`) trên cùng track. Camera hiệu chỉnh state sau LiDAR; khởi tạo, score, xác nhận và xóa track do LiDAR quyết định. Trình tự được mô tả ở §2.3 `docs/HUONG_DAN_KY_THUAT.md` và thể hiện trong runtime.

4. **Ảnh hưởng của calibration.** Calibration lệch làm projection không khớp measurement, khiến innovation/residual camera tăng và nhiều cặp bị χ² gate loại. Nếu update sai vẫn qua gate, nó có thể kéo state lệch và làm RMSE tăng. Trong alternate segment, yaw `0°→1°` làm accepted innovation median tăng `19.57→33.25 px`, camera updates giảm `946→304`, còn RMSE tăng `0.37115→0.37484 m`. Candidate innovations tính trên mọi cặp trong FOV nên không nhất thiết tăng đơn điệu; accepted residual phản ánh trực tiếp các update đã qua gate.

5. **Sensor tường minh ở frame rỗng.** `associate_and_update(manager, [], filter, sensor)` vẫn kết thúc bằng `manage_tracks(..., sensor)`. Nhờ sensor, lượt LiDAR rỗng có thể chấm miss/delete track trong FOV, trong khi lượt camera rỗng không đổi score, không tạo và không xóa track. Chính sách lifecycle khác nhau theo sensor nên cần truyền sensor tường minh kể cả khi không có measurement.

6. **Track lifecycle.** Hit LiDAR cộng `1/window` (tối đa 1); miss trong FOV trừ `1/window`. Track được xác nhận khi `score > confirmed_threshold`; một miss giảm score nhưng không hạ trạng thái confirmed. Xóa khi `P[0,0]` hoặc `P[1,1] > max_P`, hoặc confirmed có score `< delete_threshold`, hoặc track chưa confirmed có score `<=0`. Camera chỉ EKF-update, không thay đổi existence score/lifecycle.

## Bonus

- **CVAT (+3):** Export JSON ở `student/bonus/cvat_tracks_alternate.json`; annotation CVAT XML và ảnh frame 53 ở `cvat_annotations_alternate.xml` và `cvat_frames_alternate/frame_000.png`. Segment alternate với yaw extrinsic lệch `1°` có confirmed track ID 7 không ghép GT, được đánh dấu `Ghost track` (`is_ghost=true`). Đã import và kiểm tra trên CVAT Online; ảnh bằng chứng có giao diện CVAT và ghost ID 7 ở `student/bonus/cvat_import_evidence.jpg`.
- **Visualization (+3):** Ba ảnh baseline frames 31, 48, 146 tại `student/bonus/visualization_default/` thể hiện LiDAR prior, fused posterior sau camera update, GT, track ID và vector dịch chuyển. Các ảnh alternate nằm trong `visualization_alternate/`.
- **Calibration (+4):** Bảng/biểu đồ cho default và alternate segment ở `calibration_results_default.csv`, `calibration_plot_default.png`, `calibration_results_alternate.csv`, `calibration_plot_alternate.png`. Giữ cố định detections, pixel measurements và seed; chỉ thay yaw extrinsic `0°/1°/3°/5°`. Ở alternate segment, RMSE lần lượt `0.37115/0.37484/0.38830/0.39431 m`; camera updates `946/304/215/273`; accepted innovation median `19.57/33.25/22.59/20.39 px`; unmatched `4797/5439/5528/5470`; matches `1000/999/1000/999`; ghost frames `0/1/0/0`; missed GT `1534/1535/1534/1535`. Kết quả cho thấy gating loại phần lớn measurement khi extrinsic lệch, nhưng update sai còn lọt gate vẫn có thể làm RMSE xấu hơn. Mean/median/p95 innovation được lưu trong summary JSON.

## Khai báo sử dụng AI

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): ChatGPT/Codex.
- Hỗ trợ triển khai và debug Parts E–H, phân tích bonus, kiểm tra kết quả và biên tập báo cáo.
- Kiểm tra lại bằng công thức EKF/FOV/χ²/lifecycle, full test suite (`128 passed`), official Waymo compare run 199 frames và đối chiếu metrics với grade log.
