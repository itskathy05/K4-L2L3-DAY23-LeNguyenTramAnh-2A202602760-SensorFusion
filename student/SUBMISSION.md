# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

## Thông tin học viên

- Họ tên: Lê Nguyễn Trâm Anh
- MSSV: 2A202602760
- Email: kathyforwork05@gmail.com
- Link repo (fork): https://github.com/itskathy05/K4-L3-DAY23-LeNguyenTramAnh-2A202602760-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): điền sau commit cuối

## Tóm tắt kết quả official

- `fusion_mode=compare`, frames `0–198` (199 frames), segment `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, seed `0`.
- Detection: precision `0.97009`, recall `0.70040`, TP/FP/FN `519/16/222`.
- LiDAR tracking: RMSE `0.15032 m`, matches `502`, sum squared error `11.343649 m²`, ghost track-frames `0`, missed GT `239`, mean confirmed tracks `2.52261`.
- Fused tracking: RMSE `0.13587 m`, matches `502`, sum squared error `9.266812 m²`, ghost track-frames `0`, missed GT `239`, mean confirmed tracks `2.52261`.
- Cả hai mode ghép được cùng 502 cặp và có cùng ghost/miss; camera làm tổng bình phương sai số giảm `18.3%`, tương ứng RMSE giảm khoảng `0.01446 m`. Đây là cải thiện hiệu chỉnh vị trí trên các track đã ghép, không phải cải thiện recall/coverage vì detection và số matches không đổi. Detection precision/recall lần lượt là `0.97009/0.70040`; tracking precision `502/(502+0)=1.0`, coverage `502/519=0.96724` ở cả hai mode.

Lệnh chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép một-một với GT xe trong cửa sổ BEV, gate XY 2.0 m; `null` nếu không có cặp. Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, không dùng camera detector. Kết quả fused do đó không đo hiệu quả một perception system độc lập với GT.

`student/artifacts/grade_run.log` có một record cho mỗi cặp `(mode, frame)`. Invariant `matches+ghosts==confirmed` và `matches+misses==valid_gt` được kiểm tra; tổng và RMSE tái tạo từ log khớp `metrics.json`. Sáu artifacts chính gồm `metrics.json`, `metrics_lidar.json`, `metrics_fused.json`, `grade_run.log`, `grade_run_lidar.log`, `grade_run_fused.log`.

## Giải thích Parts E–H

1. **Đo EKF:** LiDAR đưa vị trí 3D `[x,y,z]` trong sensor frame vào `z`, với covariance 3×3 `R`; hàm khởi tạo đổi vị trí và covariance sang vehicle frame. Camera đưa pixel `[u,v]` vào `z` và dùng covariance ảnh 2×2 từ `sigma_cam_i²`, `sigma_cam_j²`. Mỗi sensor có hàm quan sát/Jacobian riêng nên EKF dùng `S=HPHᵀ+R` tương ứng. Xem `student/workspace/kalman.py`, `camera_fusion.py` và `track_management.py`.

2. **Mahalanobis gate:** khoảng cách Euclid không xét uncertainty và scale khác nhau giữa các sensor. Squared Mahalanobis dùng innovation covariance `S`, rồi so với phân vị χ² theo số chiều đo (`gating_threshold`). Gate loại cặp không hợp lý trước greedy assignment, ngăn một measurement ngoại lai cập nhật sai track. Với cặp ngoài FOV, phép chiếu/distance không được gọi.

3. **Track-then-fuse:** mỗi frame có một tracker chung; runtime predict một lần, ghép/update LiDAR (`AssocL`), rồi ghép/update camera (`AssocC`) trên cùng tracks. Camera refine state sau LiDAR. Track được tạo, tính score, xác nhận và xóa theo lượt LiDAR. Thứ tự này thể hiện trong `docs/HUONG_DAN_KY_THUAT.md` §2.3 và các lời gọi `associate_and_update` trong runtime; log grading lưu kết quả sau từng frame/mode.

4. **Calibration lệch:** projection camera không còn khớp, thường làm innovation pixel tăng hoặc có đuôi lớn; gate χ² sẽ tăng số camera measurement bị loại/unmatched. Update sai nhưng vẫn qua gate có thể kéo trạng thái khỏi vị trí đúng, làm RMSE tăng. Trong thí nghiệm alternate, accepted innovation median tăng từ `19.57 px` ở 0° lên `33.25 px` ở 1°; số camera update giảm `946→304` và RMSE tăng `0.37115→0.37484 m`. Các candidate innovation gồm mọi cặp trong FOV nên không nhất thiết tăng đơn điệu; median/p95 của các update được chấp nhận cho thấy residual thực tế rõ hơn.

5. **Sensor tường minh khi frame rỗng:** `associate_and_update(manager, [], filter, sensor)` vẫn phải biết lượt nào kết thúc. Runtime luôn gọi `manage_tracks(unassigned_tracks, unassigned_meas, sensor)`: lượt LiDAR rỗng có thể chấm miss/delete track trong FOV; lượt camera rỗng không được thay đổi score, tạo hoặc xóa track. Không có sensor thì lifecycle không thể phân biệt hai chính sách.

6. **Lifecycle:** hit LiDAR cộng `1/window` (tối đa 1), miss trong FOV trừ `1/window`; xác nhận khi `score > confirmed_threshold`. Miss làm giảm score nhưng không đổi trạng thái confirmed. Xóa nếu `P[0,0]` hoặc `P[1,1] > max_P`, hoặc confirmed có score `< delete_threshold`, hoặc track chưa confirmed có score `<=0`. Camera không đổi existence score/lifecycle.

## Bonus

- **CVAT (+3):** `student/bonus/cvat_tracks_alternate.json` chứa track theo frame cùng ID/state/position/size/yaw và cờ ghost; `cvat_annotations_alternate.xml` cùng ảnh `cvat_frames_alternate/frame_000.png` (Waymo frame 53) có annotation GT, LiDAR và fused. Trong segment alternate với extrinsic yaw lệch `1°`, confirmed track ID 7 không ghép GT được và được đánh dấu `Ghost track`/`is_ghost=true`; đây là thí nghiệm bonus, không phải official run (official có 0 ghost-frame). Đã import vào CVAT Online và kiểm tra trực quan; ảnh chụp màn hình có giao diện CVAT và nhãn `GHOST ID 7` nằm tại `student/bonus/cvat_import_evidence.jpg`.
- **Camera update visualization (+3):** ba ảnh baseline tại frames 31, 48, 146 trong `student/bonus/visualization_default/` vẽ LiDAR prior, camera-updated posterior, GT, ID và vector dịch chuyển; ảnh alternate nằm trong `visualization_alternate/`.
- **Calibration sensitivity (+4):** `calibration_results_default.csv`/`calibration_plot_default.png` là default segment; `calibration_results_alternate.csv`/`calibration_plot_alternate.png` là segment phụ. Yaw perturbations giữ pixel measurements/detections và seed cố định, chỉ thay extrinsic khi replay tracker. Ở alternate yaw `0/1/3/5°`, RMSE lần lượt `0.37115/0.37484/0.38830/0.39431 m`; số camera updates `946/304/215/273`; candidate innovation median `295.20/299.57/317.71/344.79 px`; accepted median `19.57/33.25/22.59/20.39 px`; unmatched measurements `4797/5439/5528/5470`; matches `1000/999/1000/999`; ghost frames `0/1/0/0`; missed GT `1534/1535/1534/1535`. Gate loại phần lớn update khi calibration lệch nhưng update sai còn lọt qua có thể làm tracking xấu đi. Summary JSON chứa mean/median/p95 innovation.

## Khai báo sử dụng AI

- Công cụ đã dùng: ChatGPT/Codex.
- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): ChatGPT/Codex.
- Dùng để triển khai và rà soát Parts E–H, hỗ trợ debug, viết công cụ phân tích/visualization/CVAT/calibration cho bonus, và soạn báo cáo. Tôi cần tự đọc lại, hiểu và chịu trách nhiệm về nội dung nộp.
- Đã kiểm tra công thức EKF, FOV, χ² gating và lifecycle trong code; chạy `pytest student/tests -q` (128 passed), chạy official Waymo `compare --seed 0` 199 frame và đối chiếu metrics với grade log.

## Checklist nộp

- [x] Parts E–H implement; full suite 128 passed, không failed/xfail.
- [x] Parts A–D giữ nguyên.
- [x] Official run `compare --seed 0`, frames 0–198.
- [x] Sáu artifacts official được runner tạo và commit; không sửa tay.
- [x] Báo cáo và khai báo AI đã điền.
- [x] Dataset, weights, `paths.yaml`, `.env` không commit.
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP` sau commit cuối.
- [ ] Đã push và nộp link repo + commit hash lên LMS.
