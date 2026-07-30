# MULTIFACTOR MLOPS — 3 BLOCKER CÒN LẠI

## Kết luận

Mã nguồn hiện đã tốt hơn, nhưng **chưa đủ trustable để dùng kết quả Optuna hoặc out-of-sample làm bằng chứng về edge**.

Ba blocker còn lại đều nằm ở correctness, không phải ở việc thiếu thêm feature, model hay hạ tầng.

---

## Blocker 1 — Chưa có một canonical pipeline duy nhất

### Hiện trạng

Repo vẫn tồn tại hai luồng:

- legacy pipeline trong `multifactor_portfolio`;
- pipeline mới trong `src/multifactor_mlops`.

Các entrypoint như train, MLflow, Optuna và backtest chưa chắc đang dùng cùng một implementation.

### Rủi ro

Có thể sửa đúng module mới nhưng lệnh chạy thực tế vẫn dùng code legacy.

Kết quả là:

- feature logic giữa train và serving khác nhau;
- portfolio construction khác nhau;
- timing contract khác nhau;
- sửa bug ở một nơi nhưng bug vẫn tồn tại ở nơi khác.

### Cách làm chuẩn

Chọn `src/multifactor_mlops` làm pipeline duy nhất:

```text
load data
→ build point-in-time features
→ build labels
→ split
→ fit
→ predict
→ construct portfolio
→ quantbt backtest
```

Yêu cầu:

1. `train.py`, MLflow, Optuna và backtest phải gọi chung pipeline này.
2. Legacy code chỉ còn là thin wrapper hoặc bị loại bỏ.
3. Không giữ hai bộ feature, label, sizing hoặc macro overlay song song.
4. Một test integration phải xác nhận mọi entrypoint gọi cùng canonical pipeline.

### Điều kiện hoàn thành

```text
[ ] Một implementation duy nhất cho feature
[ ] Một implementation duy nhất cho label
[ ] Một PortfolioConstructor duy nhất
[ ] Một MacroOverlay duy nhất
[ ] Mọi backtest đều đi qua QuantBT
```

---

## Blocker 2 — Label chưa khớp với execution

### Hiện trạng

Active pipeline vẫn có đường tạo label dạng:

```text
close D → close D+H
```

Trong khi target weight được lag một bar trước khi thực thi bằng QuantBT.

Như vậy model học một return khác với return strategy thực sự có thể giao dịch.

### Rủi ro

Đây có thể không phải look-ahead trực tiếp, nhưng gây:

- sai objective huấn luyện;
- sai ranking giữa các asset;
- đánh giá model tốt hơn hoặc tệ hơn thực tế;
- tuning model cho một target không khớp với PnL thực thi.

### Cách làm chuẩn

Chốt một timing contract duy nhất:

```text
Feature cutoff:       sau close D
Decision time:        sau close D
Execution:            open D+1
Label start:          open D+1
Label end:            open D+1+H
```

Công thức:

```text
y(D, H) = Open(D+1+H) / Open(D+1) - 1
```

Yêu cầu:

1. Prediction tạo tại D chỉ được execute từ D+1.
2. Thiếu cột `open` thì fail.
3. Không fallback âm thầm sang close-to-close.
4. Label phải lưu `label_start_time` và `label_end_time`.
5. Final fit chỉ dùng sample có:

```text
label_end_time <= training_cutoff
```

6. QuantBT integration test phải xác nhận fill thực tế đúng tại bar D+1.

### Điều kiện hoàn thành

```text
[ ] Feature chỉ dùng dữ liệu đến close D
[ ] Label bắt đầu tại open D+1
[ ] Weight được execute tại D+1
[ ] Không còn close-to-close fallback
[ ] Final train không dùng future label
```

---

## Blocker 3 — Optuna chưa phải nested walk-forward thật

### Hiện trạng

Optimizer mới có tên nested walk-forward nhưng mỗi trial vẫn có thể gọi backtest trên toàn bộ dataset hoặc trên cùng giai đoạn sau đó được dùng để báo cáo out-of-sample.

### Rủi ro

Optuna nhìn thấy test set trong quá trình chọn parameters.

Khi đó:

- best parameters bị overfit vào test;
- test period không còn là out-of-sample;
- Sharpe, CAGR và MaxDD báo cáo bị optimistic;
- không thể dùng kết quả để kết luận edge.

### Cách làm chuẩn

Dùng cấu trúc tối thiểu sau:

```text
Outer train
    ├── Inner fold 1: train → validation
    ├── Inner fold 2: train → validation
    └── Inner fold 3: train → validation

Optuna:
    chọn parameters bằng aggregate inner-validation score

Outer test:
    refit trên toàn bộ outer train
    chạy đúng một lần với best parameters
```

Yêu cầu:

1. Outer test không được truyền vào Optuna objective.
2. Inner folds phải theo thời gian, không random split.
3. Purge tối thiểu bằng label horizon.
4. Có embargo nếu samples quanh ranh giới còn overlap.
5. Mỗi trial chỉ được chấm trên inner validation.
6. Outer test chỉ được chạy sau khi study hoàn tất.
7. Không tune tiếp threshold, sizing hoặc macro overlay bằng outer test.

Pseudo-flow:

```python
for outer_fold in outer_folds:
    outer_train = data.loc[outer_fold.train]
    outer_test = data.loc[outer_fold.test]

    study = optimize_on_inner_folds(outer_train)

    best_params = study.best_params
    model = fit_on_full_outer_train(outer_train, best_params)

    result = quantbt_backtest(model, outer_test)
```

### Điều kiện hoàn thành

```text
[ ] Optuna chỉ nhận outer-train
[ ] Inner folds là walk-forward
[ ] Có purge theo label horizon
[ ] Outer test chạy đúng một lần
[ ] Outer test không dùng để chọn parameters
```

---

# Thứ tự sửa đề xuất

## Bước 1

Hợp nhất toàn bộ entrypoint vào canonical pipeline.

## Bước 2

Khóa timing contract:

```text
close D information
→ open D+1 execution
→ open D+1 đến open D+1+H label
```

## Bước 3

Viết nested walk-forward thật và cô lập outer test.

## Bước 4

Sau khi ba blocker pass:

1. xóa kết quả Optuna cũ;
2. chạy lại tuning;
3. fit lại best model;
4. chạy lại QuantBT outer-test;
5. chỉ khi đó mới đánh giá strategy còn edge hay không.

---

# Tiêu chuẩn trustable tối thiểu

```text
[ ] Một canonical pipeline duy nhất
[ ] Label và execution cùng timing contract
[ ] Optuna không nhìn outer test
[ ] Mọi backtest chạy bằng QuantBT
```

Chưa cần mở rộng thêm factor, model, provider hoặc MLOps layer trước khi bốn điều kiện này hoàn thành.
