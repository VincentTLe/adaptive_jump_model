# Tách bản thảo thành hai paper - bản V1

## Nguyên tắc viết

Hai bản mới vẫn là paper khoa học, nhưng tránh giọng “biên bản thí nghiệm”. Câu ngắn hơn, mỗi đoạn chỉ làm một việc, và kết quả kinh tế được giải thích trước khi đi vào chi tiết kiểm định. Các nhãn nội bộ như `not_supported`, run ID, hash, verifier count và lịch sử run bị vô hiệu hóa được chuyển khỏi câu chuyện chính. Chúng vẫn thuộc repository và phần reproducibility.

Phong cách hướng tới các đặc điểm tôi thích ở paper của Shu: bắt đầu từ vấn đề của nhà đầu tư, dùng chiến lược 0/1 để làm tín hiệu dễ hiểu, giải thích mô hình bằng trực giác trước phương trình, và để bảng/hình dẫn dắt phần kết quả. Đây không phải là sao chép câu chữ hay cấu trúc nguyên văn.

---

## Paper 1

### Tên hiện tại

**One Risk Measure, Fewer Trades: A Parsimonious Jump Model for U.S. Downside Protection**

### Câu chuyện trung tâm

Một Jump Model chỉ dùng downside deviation tạo ra đường đi market-or-cash tốt hơn rõ rệt ở Mỹ so với fixed JM ba feature. Nó vượt cả HMM và buy-and-hold về Sharpe, gần như khớp drawdown của HMM, nhưng giao dịch ít hơn nhiều. Loss-scale control giữ lại phần lớn improvement, nên raw loss scale không phải lời giải thích chính.

### Những gì giữ trong main paper

- Public-proxy reconstruction của economic protocol trong Shu.
- Ba benchmark: buy-and-hold, Gaussian HMM, standard three-feature JM.
- DD-only JM và three-times-DD control.
- Kết quả Mỹ là phần chính.
- Germany và Japan được giữ như external failures, không bị giấu.
- Finite-grid binding, one-state collapse và repeated sample inspection được trình bày rõ.
- Full five-model challenger suite nằm ở appendix để tránh tạo ấn tượng DD-only là giả thuyết duy nhất được thử.

### Những gì không còn nằm ở Paper 1

- Arrival, lagged và pair-balanced adaptive penalties.
- Binary adaptive recursion và amplification identity.
- Mechanism-event diagnostics, path/choice Shapley attribution.
- Separation-turnover diagnostic.
- Run IDs, commit hashes, invalidated-run history và CLI command list trong thân bài.

---

## Paper 2

### Tên hiện tại

**When Should a Jump Model Switch? Evidence-Adaptive Transition Costs with Exact Decoding**

### Câu chuyện trung tâm

Fixed JM thu cùng một transition cost dù evidence cho destination state yếu hay mạnh. Paper xây ba adaptive cost rules nhưng vẫn giữ exact dynamic programming: arrival evidence, lagged evidence và pair-balanced lagged evidence. Binary recursion cho thấy arrival rule có thể dùng cùng observation hai lần. Lagging loại bỏ direct same-day reuse và giảm candidate-path whipsaw, nhưng monthly model selection làm kết quả kinh tế không đồng nhất giữa các market.

### Đóng góp chính

- Exact time-varying transition decoder với độ phức tạp O(TK^2).
- Exact beta-zero nesting của fixed JM.
- Binary value-difference recursion.
- Same-day amplification branch identity.
- Lagged evidence rule.
- Directed-cost identity và pair-balanced construction.
- Empirical failure map: mechanism improvement không đồng nghĩa selected-strategy improvement.

### Claim được phép

Đây là methodological paper có empirical illustration. Không claim universal alpha, robust profitability hoặc cross-market superiority.

---

## Mapping từ bản thảo cũ

| Nội dung bản cũ | Paper mới |
|---|---|
| Public proxy data và walk-forward protocol | Cả hai, nhưng Paper 2 viết ngắn hơn |
| Fixed JM, HMM và 0/1 strategy | Cả hai |
| Five simple challengers | Paper 1, full table ở appendix |
| DD-only result | Paper 1, headline result |
| Three-times-DD control | Paper 1 |
| Arrival / lagged / balanced penalties | Paper 2 |
| Binary recursion và amplification | Paper 2 |
| Path-choice attribution | Paper 2 |
| Separation diagnostic | Supplement/repository; chưa cần trong main paper |
| Full forensic reproducibility ledger | Repository và supplementary material |

## Trạng thái bản V1

- Paper 1: full working draft, 11 trang, có ba figure lấy từ output hiện tại.
- Paper 2: full working draft, 10 trang, tập trung vào toán học và mechanism evidence.
- Cả hai đã compile thành PDF và đã được render kiểm tra layout.
- Đây là structural rewrite đầu tiên. Citation formatting, target-journal template, author affiliation, acknowledgements, data/code statement và supplementary appendix vẫn cần khóa trước submission.
