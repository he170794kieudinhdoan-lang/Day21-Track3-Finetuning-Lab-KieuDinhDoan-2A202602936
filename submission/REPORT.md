# Lab 21 — Evaluation Report

**Họ tên**: Kiều Đình Đoàn  **MSSV**: 2A202602936  **Ngày**: 2026-10-08
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây khớp 100% với file trong `results/`. Grader kiểm tra chéo tự động.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | Ticket CSKH tiếng Việt → JSON triage 4 trường (250 mẫu) |
| Train / val | 225 / 25 (seed 42, tỉ lệ split 90/10) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 max_steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json ghi nhận `ok: true`, `verdict: reasoning preserved`)*. 
Chat template của Qwen3.5 giữ chuẩn cặp thẻ `<think>...</think>`, không nuốt token suy luận, đảm bảo an toàn tuyệt đối khi đưa vào pipeline huấn luyện.

Về `max_length`: Thống kê token cho thấy phân vị p95 chỉ chạm 98 token (gợi ý làm tròn 256), nhưng cấu hình phần cứng tier T4 đặt `max_length = 1024` để dư dả không gian đệm, bao phủ 100% các ticket dài nhất (max 101 token) mà không bao giờ lo bị cắt cụt đuôi JSON.

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị thực nghiệm |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% token được tính loss) |
| Câu trả lời nằm trong loss | true (assert pass) |
| Câu hỏi KHÔNG nằm trong loss | true (assert pass) |

3–5 dòng đầu của đoạn được tính loss trích xuất từ `results/mask_proof.json`:

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần prompt của user và thẻ mở đầu đều được gán nhãn `-100` (được mask triệt để), loss chỉ đổ dồn vào phần sinh nội dung JSON của assistant.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3141.4 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1009.8 |
| (c) LoRA fine-tune | 0.975 | 0.611 | 1.000 | 1344.0 |

**(b) có thật sự mạnh hơn (a) không?** Có, out trình tuyệt đối:
- Baseline (a) với naive prompt hoàn toàn "tịt ngòi" ở cả target (0.000) lẫn format (0.000) vì model sinh văn bản tự do, lan man, latency bị đội lên tận 3141.4 ms.
- Baseline (b) sau khi prompt engineering tử tế đã kéo target accuracy lên 0.765, ép format chuẩn chỉ 1.000 (100% parse được JSON), đồng thời ghìm latency xuống còn 1009.8 ms.
- Tôi giữ nguyên `OPTIMIZED_PROMPT` nguyên bản (SHA: `719e74d3b6232053`), không can thiệp làm yếu prompt (b) để tạo chiến thắng ảo cho bản fine-tune. Phép đo hoàn toàn liêm chính.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6250 | 0.9750 | 391.8 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 0.0001 | 0.5373 | 0.9700 | 271.5 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | 0.0000 | 402.8 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.9400 | 469.7 | 3.86 |

### Phân tích chuyên sâu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target thực tế, `attn_only` chịu thua `correct` (0.9700 so với 0.9750). Cực kỳ thú vị là thứ tự này trái ngược 180 độ so với cột train loss: ở NB4, `attn_only` có train loss thấp hơn hẳn `correct` (0.5373 vs 0.6250). Điều này phơi bày sự thật rằng việc dồn ép rank cực đại ($r=283$) vào riêng cụm Attention chỉ khiến model học vẹt, ghi nhớ cục bộ tập train chứ không tăng tính tổng quát hóa. Gắn adapter phủ rộng toàn bộ các tầng tuyến tính (`text-linear`) với rank vừa phải ($r=16$) mới chính là đòn bẩy quyết định hiệu năng thật sự của LoRA.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Chỉ vì giảm learning rate xuống 10 lần (1e-5 — mức LR thường dùng cho Full Fine-tuning), đường loss của `wrong_lr` gần như đi ngang phẳng lì, kẹt cứng ở mức 1.5702 và target score rơi thẳng về 0.0000. Nếu chỉ nhìn vào loss cao mà không kiểm tra LR, một kỹ sư thiếu kinh nghiệm sẽ vội vã đổ lỗi cho dữ liệu rác, cho rằng bài toán quá khó, hoặc model 4B không đủ năng lực xử lý. Bản chất là do ma trận LoRA bị đóng băng trọng số gốc, cần một mức LR đủ xung lực (cỡ 1e-4 đến 2e-4) để các tham số adapter kịp hội tụ trong số bước ít ỏi.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` thể hiện khả năng ép VRAM cực gắt, giảm từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm tới 56% bộ nhớ). Tuy nhiên, cái giá phải trả không hề rẻ: thời gian train lâu nhất (469.7 giây do overhead dequantize liên tục), latency inference vọt lên 1775.6 ms, và target accuracy tụt mất 3.5% (từ 0.9750 xuống 0.9400). Số liệu thực tế hoàn toàn củng cố khuyến cáo từ tác giả Unsloth đối với thế hệ Qwen3.5: nếu GPU của bạn (như T4 16GB) đã đủ gánh fp16/bf16 LoRA thì không nên lạm dụng QLoRA kẻo rước thêm quantization error không đáng có.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.210` · `regression Δ = -0.180` · `valid_trace_rate = 0.00`

### Diễn giải phán quyết:
Cổng hồi quy đưa ra phán quyết **FAILED** dù điểm target tăng vọt từ 0.765 lên 0.975 (+21% accuracy). Nguyên nhân cốt lõi nằm ở việc năng lực suy luận tổng quát (`regression`) bị sụt giảm nghiêm trọng từ 0.7911 xuống 0.6111 (-0.180, vượt xa ngưỡng cho phép 0.020). 

Đây là minh chứng kinh điển của hiện tượng **Catastrophic Forgetting (Quên lãng thảm họa)**. Khi ta dồn ép mô hình học liên tục trên tập dữ liệu hẹp 250 mẫu chỉ toàn ticket CSKH và format JSON, mô hình bị overfit vào phong cách phản hồi ngắn gọn này và đánh mất dần tri thức thế giới chung. Kết quả FAILED này phản ánh trung thực bài toán thực tế: một model fine-tune có thể rất bá đạo trên tác vụ chuyên biệt nhưng lại trở nên "kém thông minh" ở các tác vụ đa năng nếu thiếu cơ chế bảo toàn tri thức nền.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | doi_tra, cao, chuột không dây, tich_cuc | Sai do nhầm intent khiếu nại | Đầy đủ 4 trường chuẩn 100% | ✅ FT thắng: Bắt cực chuẩn intent đổi trả dù khách khen shop hỗ trợ tốt. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhất có thể... | hoan_tien, trung_binh, ốp lưng điện thoại, tich_cuc | Đoán urgency thành cao | Đầy đủ 4 trường chuẩn 100% | ✅ FT thắng: Hiểu đúng sắc thái hoàn tiền không bị quá khẩn cấp. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | hoan_tien, thap, bình giữ nhiệt, tich_cuc | Đoán đúng urgency = thap | Đoán urgency = trung_binh | ❌ **FT thua**: Model FT bị bias bởi từ "tiền bạc" nên nâng khẩn cấp lên trung bình. |
| 4 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop... | san_pham_loi, thap, áo khoác gió, tich_cuc | Đoán đúng urgency = thap | Đoán urgency = trung_binh | ❌ **FT thua**: Thấy chữ "bị lỗi" là vội tăng urgency, bỏ quên cụm xoa dịu "khi nào tiện". |
| 5 | Cho mình hỏi, mình đặt đèn bàn LED mã đơn OD436045. Giao hàng chậm. Khi nào tiện... | van_chuyen, thap, đèn bàn LED, tich_cuc | Đoán đúng urgency = thap | Đoán urgency = trung_binh | ❌ **FT thua**: Lặp lại lỗi đánh giá sai độ khẩn cấp khi gặp cụm từ giảm nhẹ. |

**Có mẫu chung nào ở các ca FT thua không?**
Có, mẫu số chung cực kỳ rõ ràng: Toàn bộ các ca Fine-tune bị chấm điểm thấp đều gãy ở trường `urgency = thap`. Khách hàng dùng câu từ giảm nhẹ như *"Khi nào tiện"*, nhưng vì đi kèm các từ khóa phàn nàn nặng đô như *"Bị lỗi"*, *"Chưa thấy tiền"*, *"Giao hàng chậm"*, mô hình Fine-tune đã bị thiên kiến (bias) và tự động đẩy urgency lên `"trung_binh"`. Trong khi đó, Baseline (b) nhờ có prompt hướng dẫn chi tiết quy tắc ngữ nghĩa nên lại xử lý các trường hợp ngoại lệ này tốt hơn.

---

## 7. Kết luận & điều tôi học được

**Kết luận chuyên môn:**  
Có nên deploy bản fine-tune này lên production không? Câu trả lời là: **Chưa thể deploy thẳng thừng nếu chưa có biện pháp bọc lót.** 

Mặc dù bản Fine-tune đạt Target Accuracy rất cao (0.975 so với 0.765 của Prompting) và tốc độ ổn định, nhưng việc dính phán quyết FAILED do năng lực tổng quát tụt 18% là một rủi ro lớn nếu triển khai làm chatbot đa năng. Nếu hệ thống kiến trúc theo dạng Microservices — tách riêng một worker backend chỉ chuyên ăn text ticket và nhả JSON triage — thì bản LoRA này hoàn toàn phát huy tác dụng vượt trội về độ ổn định format và độ chính xác phân loại. Tuy nhiên, nếu model này phải tương tác trực tiếp với người dùng cuối, hiện tượng quên lãng thảm họa sẽ biến nó thành một bot ngáo ngơ khi gặp các câu hỏi nằm ngoài nghiệp vụ CSKH.

Đòn bẩy thực sự làm nên thành công của bài lab này không nằm ở việc cố kéo rank LoRA lên thật cao, mà nằm ở bộ ba: **Vị trí gắn adapter toàn diện (`text-linear`)**, **Learning rate chuẩn thang LoRA ($10^{-4}$)**, và **Loss masking chuẩn chỉ (`assistant-only`)**.

**Ba điều tôi học được:**
1. **Đừng để Train Loss đánh lừa thị giác**: Train loss thấp lè tè đôi khi chỉ là model đang học vẹt. Bằng chứng sống là `attn_only` có loss 0.5373 nhưng target vẫn thua `correct` có loss 0.6250. Đánh giá chất lượng model bắt buộc phải dùng Task Metric thực chiến trên tập holdout.
2. **Loss Masking là lằn ranh sinh tử**: Nếu tính loss cả vào prompt, model sẽ biến thành con vẹt copy câu hỏi. Mask đúng câu trả lời của assistant chính là gốc rễ giúp mô hình tập trung 100% năng lượng vào cấu trúc JSON cần học.
3. **Prompting là tấm gương soi liêm chính**: Một baseline (b) được prompt chỉn chu là tiêu chuẩn vàng để biết mình có thực sự cần fine-tune hay không. Nếu không có baseline (b), ta sẽ dễ dàng ngộ nhận rằng LoRA đang tạo nên phép màu, trong khi một vài dòng system prompt khéo léo đã có thể giải quyết 80% bài toán.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Trộn thêm 3–5% dữ liệu đàm thoại tổng quát (replay data) vào tập huấn luyện để chặn đứng đà tụt dốc của cổng hồi quy `regression`, đưa phán quyết về xanh `PASSED`.
- Bổ sung thêm các mẫu few-shot đặc thù cho cụm từ giảm nhẹ như *"Khi nào tiện"* vào dữ liệu train để khắc phục triệt để lỗi bias trường `urgency`.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/tesfwefew/qwen3.5-4b-cskh-lora
