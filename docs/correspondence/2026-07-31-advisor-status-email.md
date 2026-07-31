# Email gửi thầy trước buổi họp 31/07/2026

---

Kính gửi thầy,

Trước buổi họp chiều nay, em xin tóm tắt toàn bộ tiến độ tái lập paper Shu, Yu & Mulvey (2024), "Downside risk reduction using regime-switching signals: a statistical jump model approach" (J. Asset Management; arXiv:2402.05272), tình trạng hiện tại của mô hình JM, và khó khăn trung tâm — lưới tham số λ không được công bố.

## 1. Dữ liệu thay thế (paper dùng Bloomberg + GFD, em tái dựng từ nguồn miễn phí)

- **S&P 500 Total Return**: chuỗi ^SP500TR chính thức từ 1988, nối với đoạn tái dựng trước 1988; một lỗi ghép chuỗi (mất phiên 04/01/1988) đã được phát hiện và sửa, có test hồi quy bảo vệ.
- **DAX**: bản chất là chỉ số total return; đoạn trước 1988 theo dòng backcast Stehle, đối chiếu chéo với dữ liệu tháng của OECD.
- **Nikkei 225 TR**: chỉ số TR chính thức chỉ tồn tại từ 28/12/1979 (và chỉ được phát hành từ 2012) — nghĩa là giai đoạn 1970-79 của chính tác giả cũng phải là chuỗi tự dựng; em dựng từ JST Macrohistory + chuỗi chính thức + cầu nối cho lỗ hổng 2020-22.
- **Lãi suất phi rủi ro**: T-bill 3 tháng Mỹ (FRED DTB3, đã đối chiếu byte-identical với bản tải tay), thang Bundesbank cho Đức, thang IMF/BoJ cho Nhật.
- Mọi run đều "niêm phong": config + manifest dữ liệu + git commit được băm; chạy lại cho sai lệch đúng 0.0.

## 2. HMM baseline — coi như tái lập thành công

Trên run niêm phong: **7/8 ô của Table 4 nằm trong dung sai 0.05 ở CẢ BA thị trường** (delay 1). Nhiều ô khớp gần tuyệt đối (volatility Đức lệch 0.00004). Ô duy nhất trượt là turnover (em: 1.79/2.26/3.14 so với 1.41/2.46/2.90), và nguyên nhân đã được truy tận gốc: (i) tham số làm mượt k cũng được chọn từ một lưới ứng viên không công bố; (ii) Sharpe trên tập validation gần như phẳng theo k — điều mà chính dòng tài liệu họ trích (luận án Nystrup 2018) đã ghi nhận — nên phép chọn hyperparameter không ổn định về turnover theo đúng nghĩa nguyên lý; (iii) slide hội nghị của chính tác giả in "96 lần đổi trạng thái, 27.8% ngày bear" cho US — em ra 122 lần / 27.95%: chiếm hữu trạng thái khớp, chỉ thừa các cú lật ngắn.

## 3. JM — phần chưa khớp, và bằng chứng rằng hệ thống của em không phải nguyên nhân

Phép kiểm quyết định: **shading regime trong Figure 5 của paper là dữ liệu vector không mất mát — em trích được đúng chuỗi trạng thái từng ngày mà tác giả đã chạy** (khớp chính xác các con số họ in trên từng panel: 30/116/48 lần đổi, 19.7%/15.7%/25.3% bear). Áp chuỗi CỦA HỌ lên dữ liệu CỦA EM: **tái tạo 23/24 ô hàng JM của Table 4** (turnover lệch 0.001/0.030/0.005; Sharpe lệch 0.008/0.003/0.023). Tức là dữ liệu, kế toán phí, quy ước giao dịch của em đều đúng — toàn bộ phần chưa khớp nằm ở việc *tự sinh lại chuỗi trạng thái đó*, vốn phụ thuộc hai thứ paper không công bố: lưới ứng viên λ và công thức chuẩn hoá đặc trưng.

Chạy độc lập của em hiện đạt 4/8 (US), 3/8 (DE), 3/8 (JP) — và em đã chạy chuỗi thí nghiệm đông cứng (đăng ký câu hỏi + tiêu chí trước khi chạy, có kiểm toán đối kháng bằng agent độc lập) để đặc tả khoảng cách này.

## 4. Tại sao lưới λ quan trọng đến vậy (giải thích ngắn cơ chế)

Mô hình JM về bản chất là k-means theo thời gian có phạt: mỗi ngày được gán vào trạng thái bull/bear sao cho tổng "khoảng cách đặc trưng + λ × (số lần đổi trạng thái)" nhỏ nhất. **λ là núm điều chỉnh độ lì của chuỗi trạng thái**: λ=0 cho ~9.7 lần đổi/năm, λ=150 chỉ còn ~0.4 (số của chính Table 3). Chiến lược thì mỗi tháng chọn lại λ̂ **từ một lưới ứng viên** bằng Sharpe trên cửa sổ validation 8 năm. Vậy lưới quyết định "menu độ lì" mà phép chọn được phép dùng — và vì Sharpe validation gần phẳng giữa các ứng viên, phép chọn rất nhạy với thành phần lưới. Trên **cùng một dữ liệu, cùng một code**, chỉ đổi lưới: Sharpe DE của em chạy từ 0.13 tới 0.49, turnover US từ 0.24 tới 0.82. Lưới không phải chi tiết kỹ thuật — nó là tham số chi phối kết quả, và paper chỉ viết đúng một câu "chọn từ một dải giá trị ứng viên" mà không nêu dải nào.

## 5. Lưới λ của những nơi khác (không ai biết lưới thật)

- **Chính tác giả, arXiv v1**: từng công bố lưới {10, 22, 50, 100, 220, 500, 1000} — rồi RÚT ở bản v3 (đổi protocol) và không công bố lưới mới. Luận án PhD Princeton 2025 của Shu (em đã tải toàn văn) cũng chỉ ghi "a list of candidate jump penalties".
- **Li, Chen, Tao & Ji 2025** (Mathematics 13(17):2837, trích dẫn paper): tự đặt {0, 5, 10, 25, 50, 100}, chọn bằng information criteria — và λ dồn về biên trên ở cả 12 tài sản.
- **Nhóm sinh viên Bocconi** (B&S Capital Markets): tự đặt {0, 5, ..., 150}, chỉ chạy S&P, cửa sổ khác — không so được với Table 4.
- **Notebook cộng đồng (HackMD)**: tự đặt {0, 0.1, 1, 10, 100}, chạy DJIA.
- Em đã quét bằng máy **toàn bộ 14 paper trích dẫn** trên Semantic Scholar: **chưa có bất kỳ bản tái lập độc lập nào của protocol 3 thị trường công bố kết quả khớp Table 4** — mọi bản cài độc lập đều phải tự bịa lưới vì lưới thật không tồn tại ở đâu công khai.

## 6. Truy ngược và tìm kiếm cạn kiệt — kết quả mới nhất (đêm qua/sáng nay)

- Ước lượng lưới từ chính lựa chọn hàng tháng của tác giả (đọc từ Figure 5): lưới US của họ nằm quanh dải **[10, 100], trọng tâm ~35**.
- **Quét cạn 6.474.511 tổ hợp lưới** (mọi tập con cỡ 2-8 của 29 giá trị λ có nguồn gốc), 9 cánh (3 thị trường × delay 1/5/10, delay dài chấm theo Table 5), evaluator được kiểm ở độ chính xác máy và mọi kết quả nêu tên đều chấm lại bằng pipeline gốc:
  - **US: tìm được lưới khớp Shu ở TOÀN BỘ 14 ô** (8 ô Table 4 delay-1 + 3 ô Table 5 ở mỗi delay 5/10). Ví dụ lưới {0, 21.5, 70}: lệch tệ nhất 0.011 ở delay 1. Có 36.657 lưới như vậy.
  - **DE/JP: trong menu 29 giá trị, không lưới nào đạt đủ 8 ô delay-1** — và phân tích biên chỉ đúng ô chặn: DE là turnover (chuỗi giữ được 7 ô kia thì giao dịch tối đa 1.47 so với đích 1.70 — ràng buộc liên hợp, không phải thiếu giá trị λ vì phổ turnover toàn cục lên tới 4.6); JP là leverage (trần 0.74 so với đích 0.75, và chỉ 0.64 khi 7 ô kia đạt — chuỗi của em cố hữu "bearish" hơn của họ). Em nhấn mạnh phát biểu này có phạm vi: *trong menu hiện tại*; bước tiếp theo em sẽ thử mở rộng lưới sang giá trị thực/dày hơn cho DE/JP để loại trừ khả năng menu hẹp, dù bằng chứng biên cho thấy vấn đề nằm ở hình dạng chuỗi trạng thái (từ hình học chuẩn hoá đặc trưng — paper cũng chỉ viết một chữ "standardized") hơn là ở độ phủ của λ.

## 7. Vì sao chưa chạy các thí nghiệm mở rộng JM

Các hướng mở rộng đã thiết kế từ trước (capped gap, two-day confirmation, semi-Markov dwell cost) em **chưa chạy**, vì baseline JM hiệu chỉnh chưa đạt mức khớp Shu ở cả ba thị trường — chạy mở rộng trên baseline lệch sẽ không quy chiếu được. Quyết định đang chờ: có "reseal" baseline với lưới hiệu chuẩn (US đã có lưới 14/14; DE/JP dùng lưới tốt nhất 13/14 kèm ô chặn ghi rõ) — dán nhãn minh bạch là hiệu chuẩn chứ không phải lưới của tác giả — để phần mở rộng có mốc so sánh sạch.

## Link tham khảo

- Paper (v3): https://arxiv.org/abs/2402.05272 · Bản v1 có lưới bị rút: https://arxiv.org/abs/2402.05272v1
- Luận án PhD của Shu (toàn văn, miễn phí): https://dataspace.princeton.edu/handle/88435/dsp01g158bm716
- Slide Wolfe Research của Shu (in các con số 96 shifts / 30 shifts): https://drive.google.com/file/d/1-8a9GzfyDELUIq0rq7NF2iqmyMikCmGr/view
- Package chính thức (không chứa lưới/CV): https://github.com/Yizhan-Oliver-Shu/jump-models
- Li et al. 2025: https://www.mdpi.com/2227-7390/13/17/2837
- Bocconi Students Capital Markets: https://www.bscapitalmarkets.com/statistical-jump-models-for-regime-switching.html
- Notebook HackMD (DJIA): https://hackmd.io/@e41406/HkUKkKTpR

Toàn bộ thí nghiệm, sổ đăng ký và kiểm toán nằm trên branch `cleanup/research-protocol` (registry: `research/experiment_registry.jsonl`; sổ: `docs/audit/`). Em sẵn sàng trình chi tiết bất kỳ mục nào trong buổi họp.

Em cảm ơn thầy.
