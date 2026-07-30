# MULTIFACTOR MLOPS — CÁC TỒN ĐỌNG VÀ KẾ HOẠCH SỬA TỐI THIỂU

## 1. Mục tiêu

Giữ nguyên thesis và bộ feature hiện tại. Chưa mở rộng thêm factor, model hoặc data source.

Mục tiêu duy nhất của đợt sửa này là làm cho pipeline:

- không look-ahead bias;
- train, tuning và backtest tách biệt đúng;
- signal, sizing và execution nhất quán;
- toàn bộ backtest chạy bằng `quantbt`;
- dữ liệu từ `trading-historical-data` được align đúng theo thời điểm có thể sử dụng;
- kết quả có thể tái lập từ một cấu hình thống nhất.

---

## 2. Đánh giá trạng thái hiện tại

Nhánh mới đã cải thiện hai điểm quan trọng:

1. Split lỗi không còn tự động rơi về full-sample.
2. Backtest chính đã bắt đầu đi qua `QuantBTRunner` và có lag execution.

Tuy nhiên pipeline vẫn chưa đủ trustable vì code mới và code legacy còn chạy song song. Một số lỗi correctness vẫn có thể làm sai toàn bộ kết quả.

---

# 3. Các tồn đọng bắt buộc phải sửa

## P0.1 — Chưa có một pipeline duy nhất

Hiện tồn tại đồng thời:

- pipeline legacy trong `multifactor_portfolio`;
- pipeline mới trong `src/multifactor_mlops`.

CLI, MLflow, Optuna, train và serving chưa chắc đang gọi cùng một implementation.

### Rủi ro

Có thể sửa đúng module mới nhưng lệnh chạy thực tế vẫn dùng module cũ.

### Giải pháp

Chọn `src/multifactor_mlops` làm pipeline chuẩn.

Các entrypoint sau phải gọi cùng một pipeline:

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

Code legacy chỉ được:

- xóa; hoặc
- giữ làm thin wrapper gọi sang pipeline mới.

Không được duy trì hai bộ logic song song.

---

## P0.2 — Vẫn còn `bfill`

Một số module macro và serving vẫn dùng:

```python
ffill().bfill()
```

### Rủi ro

`bfill` lấy dữ liệu tương lai điền ngược về quá khứ, gây look-ahead bias trực tiếp.

### Giải pháp

Cấm toàn bộ `bfill` trong pipeline research, train, backtest và serving.

Quy tắc:

```text
- chỉ forward-fill từ quan sát đã xuất hiện;
- không có dữ liệu quá khứ thì giữ NaN;
- loại các hàng chưa đủ warm-up;
- không dùng dữ liệu tương lai để lấp đầu chuỗi.
```

Thêm một test scan toàn repo để fail nếu xuất hiện `.bfill(` trong source code đang chạy.

---

## P0.3 — Timing của feature, label và execution chưa thống nhất

Pipeline mới hướng tới label `next-open → future-open`, nhưng pipeline cũ vẫn có đường tính `close → future-close`.

Ngoài ra, một số module có thể fallback sang close nếu thiếu open.

### Rủi ro

Model có thể được train bằng một timing contract nhưng backtest lại thực thi theo timing khác.

### Giải pháp

Chốt một contract duy nhất:

```text
Feature cutoff:       hết bar D
Decision time:        sau close D
Order execution:      open D+1
Label start:          open D+1
Label end:            open D+1+H
```

Yêu cầu:

- thiếu cột `open` thì fail;
- không fallback âm thầm sang close-to-close;
- prediction tại D chỉ được dùng từ D+1;
- integration test phải xác nhận QuantBT fill tại đúng bar.

---

## P0.4 — Serving có thể đổi short thành long

Risk weight hiện đã mang dấu:

```text
long  → weight dương
short → weight âm
```

Nhưng serving vẫn có đường nhân signed signal với signed risk weight:

```text
-1 × -0.5 = +0.5
```

### Rủi ro

Vị thế short bị đảo thành long.

### Giải pháp

Tạo một `PortfolioConstructor` duy nhất dùng cho:

- validation;
- backtest;
- MLflow model;
- paper/live inference.

Output cuối cùng phải là signed target weight. Không nhân dấu thêm lần thứ hai.

Thêm test bắt buộc:

```text
short prediction → final weight < 0
long prediction  → final weight > 0
```

---

## P0.5 — Macro overlay giữa backtest và serving khác nhau

Backtest và serving hiện chưa dùng cùng công thức macro overlay:

- rolling window khác nhau;
- threshold/sigmoid khác nhau;
- multiplier khác nhau.

### Rủi ro

Portfolio chạy thật không tái tạo được portfolio đã backtest.

### Giải pháp

Tách macro overlay thành một pure component duy nhất:

```python
adjusted_weights = macro_overlay.transform(
    decision_time=decision_time,
    macro_features=macro_features,
    raw_weights=raw_weights,
)
```

Cùng một class, cùng parameters và cùng output phải được dùng ở train validation, backtest và serving.

---

## P0.6 — Optuna vẫn có nguy cơ tối ưu trên test set

Optuna hiện vẫn có thể dùng metric của giai đoạn được gọi là out-of-sample để chọn best parameters, rồi báo cáo lại chính giai đoạn đó.

### Rủi ro

Kết quả bị test-set overfitting. Best trial không còn là bằng chứng out-of-sample.

### Giải pháp tối thiểu

Dùng hai tầng:

```text
Outer train
    └── inner walk-forward folds cho Optuna

Outer test
    └── chỉ chạy một lần sau khi chọn best parameters
```

Quy tắc:

- Optuna không được nhìn outer test;
- purge ít nhất bằng label horizon;
- best parameters được chọn bằng trung bình inner validation;
- outer test không được dùng lại để tune threshold, feature hoặc sizing.

Không cần xây framework phức tạp hơn mức này.

---

## P0.7 — Final fit có thể dùng label nằm sau cutoff

Pipeline có thể tạo forward labels trên toàn bộ data rồi chỉ lọc theo `decision_time <= cutoff`.

### Rủi ro

Một sample tại cutoff có thể dùng giá sau cutoff để tạo target.

### Giải pháp

Mỗi label phải có:

```text
decision_time
label_start_time
label_end_time
```

Final train chỉ được giữ sample thỏa:

```python
label_end_time <= training_cutoff
```

Không chỉ kiểm tra `decision_time`.

---

# 4. Các tồn đọng quan trọng tiếp theo

## P1.1 — Universe phải point-in-time

Không được chọn top thanh khoản bằng average turnover trên toàn bộ train/test period.

### Giải pháp

Tại mỗi rebalance date D:

```text
liquidity window = dữ liệu đến D-1
universe at D    = top N theo window quá khứ
```

Không dùng thanh khoản tương lai để quyết định symbol có mặt trong quá khứ.

---

## P1.2 — Align dữ liệu phải theo thời điểm có thể sử dụng

`trading-historical-data` được xem là nguồn dữ liệu chính và đáng tin. Không cần thêm hệ thống provenance phức tạp trong giai đoạn này.

Điều bắt buộc là align đúng.

### Quy tắc align

#### Price và volume

- Chuẩn hóa timezone trước khi merge.
- Xác định timestamp là bar open hay bar close.
- Feature của bar D chỉ khả dụng sau khi bar D hoàn tất.
- Không forward-fill OHLCV qua ngày không giao dịch.
- Không bù missing price bằng dữ liệu tương lai.

#### Funding, open interest và market data khác

- Chỉ merge observation có timestamp nhỏ hơn hoặc bằng decision cutoff.
- Dùng backward `merge_asof`.
- Không dùng nearest merge.
- Không dùng daily aggregate chứa dữ liệu xảy ra sau decision time.

#### Macro data

- Align theo thời điểm dữ liệu thực sự có thể quan sát.
- Chỉ forward-fill sau lần công bố gần nhất.
- Không `bfill`.
- Nếu chưa xác định được thời điểm công bố hợp lệ thì feature đó phải bị loại khỏi sample, không được đoán.

Alternative source chỉ dùng khi dữ liệu chính chưa update, nhưng phải áp dụng cùng timing contract và không được ghi đè lịch sử đã có một cách âm thầm.

---

## P1.3 — `parameters.json` chưa đồng bộ hoàn toàn

Một số model parameters vẫn có thể nằm nhầm trong section `features`, trong khi code mới đọc từ `model`.

### Rủi ro

Cấu hình bị bỏ qua và pipeline âm thầm dùng default.

### Giải pháp

Giữ một file cấu hình duy nhất:

```json
{
  "data": {},
  "features": {},
  "labels": {},
  "universe": {},
  "model": {},
  "training": {},
  "optimization": {},
  "portfolio": {},
  "execution": {},
  "backtest": {}
}
```

Config loader phải:

- fail khi có key không hợp lệ;
- fail khi thiếu field bắt buộc;
- không âm thầm dùng default cho key viết sai;
- được dùng chung bởi train, Optuna và backtest.

---

## P1.4 — Funding feature và funding accounting phải nhất quán

Nếu funding được dùng làm factor và strategy trade perpetual futures, QuantBT phải account funding cashflow.

### Giải pháp

- truyền historical funding series vào QuantBT;
- align đúng funding timestamp;
- không dùng constant funding rate thay cho lịch sử nếu đã có dữ liệu;
- nếu chưa account được funding thì kết quả phải được đánh dấu non-certified.

---

# 5. Luồng implementation tối thiểu nên giữ

```text
1. Load dữ liệu từ trading-historical-data
2. Chuẩn hóa timezone và bar semantics
3. Build point-in-time universe
4. Build feature chỉ từ dữ liệu <= decision cutoff
5. Build next-open forward label
6. Tạo walk-forward split có purge
7. Fit model trên train
8. Predict trên validation/test
9. Dùng một PortfolioConstructor duy nhất
10. Áp dụng một MacroOverlay duy nhất
11. Lag target weight sang bar execution
12. Backtest duy nhất bằng QuantBT
13. Optuna chỉ tune trên inner validation
14. Đánh giá một lần trên outer test
15. Final fit chỉ dùng label_end_time <= cutoff
```

---

# 6. Bộ test tối thiểu bắt buộc

Không cần mở rộng test framework quá nhiều. Chỉ cần khóa các lỗi có thể làm sai backtest:

```text
[ ] Không tồn tại bfill trong active source
[ ] Feature tại D không đổi khi sửa dữ liệu sau D
[ ] Label tại D bắt đầu từ open D+1
[ ] Prediction tại D không được execute tại D
[ ] Short signal luôn tạo weight âm
[ ] Backtest và serving tạo cùng weight
[ ] Macro overlay backtest và serving giống nhau
[ ] Universe tại D không đổi khi sửa liquidity sau D
[ ] Optuna không truy cập outer test
[ ] Final fit không dùng label_end_time sau cutoff
[ ] Tất cả backtest đi qua QuantBT
```

---

# 7. Thứ tự sửa ngắn gọn

## Bước 1 — Hợp nhất pipeline

- Chọn pipeline mới làm canonical.
- Nối CLI, MLflow, Optuna và backtest vào pipeline đó.
- Ngừng chạy logic legacy độc lập.

## Bước 2 — Khóa leakage và parity

- Xóa `bfill`.
- Thống nhất timing contract.
- Sửa short sign.
- Dùng chung portfolio constructor và macro overlay.
- Sửa final-fit cutoff.

## Bước 3 — Khóa evaluation

- Inner walk-forward cho Optuna.
- Outer test untouched.
- Universe point-in-time.
- Align funding và macro đúng cutoff.

## Bước 4 — Rerun

Chỉ sau khi các test tối thiểu pass mới:

- chạy lại Optuna;
- chạy lại QuantBT backtest;
- so sánh kết quả mới với kết quả cũ;
- kết luận strategy còn edge hay không.

---

# 8. Điều kiện để coi chiến lược là trustable

Chiến lược đạt mức trustable khi:

```text
- một pipeline duy nhất;
- không còn backward fill;
- feature/label/execution cùng timing contract;
- train, backtest và serving parity;
- Optuna không tune trên test;
- universe và macro đều point-in-time;
- QuantBT là backtest engine duy nhất;
- final fit không dùng future labels.
```

Không cần mở rộng thêm factor, model, provider hoặc MLOps layer trước khi đạt các điều kiện trên.
