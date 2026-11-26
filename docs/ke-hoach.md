# Kế hoạch thực hiện chủ đề 8 · Dịch máy Anh–Việt

**Câu hỏi nghiên cứu.** Trên cùng các câu tiếng Anh từ TED, LSTM có attention và Transformer nhỏ khác nhau thế nào về chất lượng bản dịch tiếng Việt? Kết quả thay đổi ra sao khi đổi đơn vị token hoặc thuật toán giải mã? Fine-tune một mô hình dịch đã học trước mang lại mức BLEU nào trên cùng phép đo?

Đầu ra của hệ thống gồm bản dịch và bản đồ attention có nhãn token Anh–Việt. Nhóm dùng IWSLT 2015, đo corpus BLEU, đọc lỗi trên từng câu và lưu đủ thông tin để người khác tính lại bảng kết quả.

## Phân công và bàn giao

| Phụ trách | Công việc chính | Sản phẩm bàn giao và tiêu chí nghiệm thu |
|---|---|---|
| TV1 · Dữ liệu, tokenizer, đánh giá | Kiểm tra corpus, chuẩn hóa, xây ba bộ mã hóa, thống nhất định dạng dự đoán, tính BLEU, tổng hợp lỗi | Hàm đọc cặp có ID ổn định; vocab và cấu hình từng ngôn ngữ; bảng thống kê độ dài/OOV; notebook kiểm đủ ID trước khi chấm. Bàn giao cho TV2 và TV3 cùng ví dụ encode/decode. |
| TV2 · Seq2seq attention | Tự viết encoder/decoder LSTM, Luong attention, teacher forcing; phụ trách giao diện giải mã chung và phối hợp chạy EnViT5 | Kiểm tra attention bỏ PAD; thử overfit một batch; checkpoint, log và hai bản dịch trên mỗi tokenizer; hình attention có nhãn trục. Bàn giao bộ kiểm thử decoder cho TV3. |
| TV3 · Transformer | Tự viết Transformer encoder–decoder nhỏ; tích hợp greedy/beam; phụ trách cấu hình fine-tune EnViT5 | Kiểm tra causal mask, cross-attention và shape theo head; thử overfit một batch; checkpoint, cấu hình, dự đoán và attention của Transformer. Cùng TV2 kiểm tra tiền tố, checkpoint và đầu ra EnViT5. |

TV1 viết phần dữ liệu và kết quả, TV2 viết phần LSTM và phân tích attention, TV3 viết phần Transformer và EnViT5. Cả nhóm đối chiếu lại bảng bằng tệp dự đoán, thống nhất kết luận và ghi họ tên, mã sinh viên, tỉ lệ đóng góp thực tế khi nộp.

## Chuẩn bị corpus và token hóa

1. Chạy script tải dữ liệu; lưu số dòng gốc và số cặp giữ lại. Dùng train để học tham số, tst2012 để chọn cấu hình, tst2013 để đánh giá sau khi chốt. Giữ số dòng nguồn làm ID. Cùng bỏ hai vế nếu cặp train có một vế rỗng; ghi nhận 151 cặp như vậy. Kiểm tra cặp trùng trong và giữa các split, ghi số lượng và quyết định xử lý trước khi huấn luyện.
2. Giải mã HTML entity, chuẩn hóa Unicode NFC và khoảng trắng, giữ hoa thường, dấu câu và dấu tiếng Việt. Ghi phân vị độ dài hai phía, tỷ lệ câu vượt giới hạn và một ví dụ trước/sau. Cố định cùng danh sách ID huấn luyện cho sáu cấu hình tự cài; ghi tỷ lệ cắt theo tokenizer.
3. Học vocab nguồn và đích riêng trên train. Với BPE và Unigram dùng hai thuật toán trong cùng thư viện SentencePiece, cùng cỡ vocab mục tiêu 8.000 mỗi phía và cùng chuẩn hóa. Với mức từ, dùng token tiếng Anh đã cách sẵn trong corpus, tách từ ghép tiếng Việt bằng underthesea, rồi lấy các từ thường gặp nhất trong cùng ngân sách vocab. Báo cáo cỡ thực, tỷ lệ UNK và số token/câu trên validation.
4. Thống nhất PAD=0, UNK=1, BOS=2, EOS=3 cho mô hình tự cài. Thử vòng mã hóa–giải mã trên câu có tên riêng, từ ghép và số. Khi xuất bản dịch, bỏ token đặc biệt, dùng hàm decode của từng tokenizer và chuyển dấu nối từ ghép nội bộ về khoảng trắng. TV1 kiểm tra bản dịch và tham chiếu dùng chung quy ước văn bản.

## Xây dựng và huấn luyện

1. **LSTM.** Bắt đầu với embedding/hidden 256, hai tầng. Encoder trả hidden theo từng vị trí; decoder tính Luong dot attention giữa trạng thái hiện tại và các vị trí nguồn hợp lệ. Ghép context vào đầu ra để dự đoán token kế tiếp. Huấn luyện bằng teacher forcing, cross-entropy bỏ PAD và gradient clipping. Xác nhận attention mỗi bước có tổng bằng 1 và bằng 0 trên PAD.
2. **Transformer.** Bắt đầu với ba tầng encoder và ba tầng decoder, chiều 256, bốn head, FFN 1.024. Tự cài sinusoidal positional encoding, scaled dot-product attention, multi-head, FFN, residual và layer norm. Kiểm tra thay đổi token tương lai không làm đổi logits quá khứ khi tắt dropout. Lưu cross-attention của decoder để so vị trí nguồn và đích.
3. **EnViT5.** Dùng checkpoint dịch của VietAI cùng tokenizer gốc. Tạo input có tiền tố `en:`, nhãn có tiền tố `vi:`, đệm nhãn bằng -100. Điểm xuất phát là learning rate 3e-5, batch 4, tích lũy gradient 8 bước và tối đa 128 token mỗi phía. Fine-tune trên train, chọn checkpoint theo validation loss. Ghi revision, batch thực và số câu bị cắt; bỏ tiền tố ngôn ngữ ở đầu bản dịch trước khi chấm. Diễn giải riêng lợi thế dữ liệu huấn luyện trước. [Model card EnViT5](https://huggingface.co/VietAI/envit5-translation)
4. **Kiểm tra trước khi chạy đủ.** Hai mô hình tự cài phải giảm được loss trên một batch nhỏ, giải mã đến EOS hoặc giới hạn chiều dài và nạp lại checkpoint cho cùng kết quả khi tắt dropout. Chốt seed 42, ngân sách số lượt qua train, tiêu chí chọn checkpoint và không gian dò learning rate trên validation. Giữ ngân sách nhất quán khi thay tokenizer trong từng kiến trúc. Lưu số tham số, optimizer, batch, loss, validation BLEU, phiên bản thư viện, phần cứng và log TensorBoard cho mỗi lần chạy.

## Ma trận thí nghiệm và điều kiện so sánh

| Nhánh | Tokenizer | Checkpoint cần huấn luyện | Giải mã mỗi checkpoint | Số bộ dự đoán test |
|---|---|---:|---|---:|
| LSTM + Luong | Mức từ, BPE, Unigram | 3 | Greedy và beam | 6 |
| Transformer nhỏ | Mức từ, BPE, Unigram | 3 | Greedy và beam | 6 |
| EnViT5 fine-tune | Tokenizer gốc | 1 | Greedy và beam | 2 |

Greedy lấy token xác suất cao nhất mỗi bước. Beam khởi đầu với độ rộng 4, cộng log-probability, lưu riêng giả thuyết đã kết thúc và chọn câu hoàn chỉnh theo quy tắc chuẩn hóa độ dài đã chốt trên validation. Dùng cùng checkpoint, bộ câu, quy ước EOS và giới hạn token đầu ra cho hai decoder của từng cấu hình; tắt sampling. Kiểm tra beam rộng 1 trả kết quả greedy với cùng quy tắc dừng. Báo cáo cả tỷ lệ chạm giới hạn vì cùng số token có thể ứng với độ dài văn bản khác nhau.

So kiến trúc bằng hai hàng cùng tokenizer và decoder; so tokenizer trong cùng kiến trúc và decoder; so decoder trên cùng checkpoint. Ghi cả chênh lệch BLEU và tốc độ câu/giây với cùng thiết bị, batch suy luận và cách đo. Nếu đủ tài nguyên, lặp lại các cấu hình chính với nhiều seed và báo trung bình, độ lệch chuẩn.

## Chấm điểm, phân tích và sản phẩm nộp

- TV1 khóa cấu hình trước khi chấm test. Mỗi lượt xuất một JSONL chứa toàn bộ 1.268 ID theo thứ tự gốc và trường `prediction`. Notebook 05 ghép đúng câu tham chiếu, tính SacreBLEU ở mức corpus, giữ hoa thường, tokenizer 13a, ghi signature. Bảng chính ghi mô hình, tokenizer, decoder, seed, BLEU, chênh lệch so với greedy, tốc độ và checkpoint.
- Chọn trước các ID validation đại diện câu ngắn/dài, từ hiếm, tên riêng, số và phủ định. Đặt câu nguồn, tham chiếu, hai bản dịch và attention cạnh nhau. Với Transformer ghi rõ tầng/head; với LSTM ghi rõ bước decoder. Phân loại thiếu ý, dịch sai, lặp từ và thứ tự từ. Dẫn ví dụ mà beam tốt hơn và ví dụ beam vẫn sai nếu quan sát được; nêu giới hạn của BLEU một tham chiếu.
- Proposal ACL tối đa 2 trang trình bày bài toán, corpus, baseline, cách đánh giá và phân công. Milestone tối đa 4 trang có kết quả baseline thật, phần đã hoàn thành, khó khăn và phần còn lại. Đọc và trích dẫn các công trình gốc về attention, Transformer và EnViT5.
- Báo cáo cuối tiếng Việt theo ACL bản final, A4 hai cột, tối đa 8 trang nội dung. Gồm abstract 150–250 từ có kết quả định lượng, giới thiệu, nghiên cứu liên quan, phương pháp, thí nghiệm, phân tích, kết luận; bổ sung Limitations, đóng góp thành viên và tài liệu tham khảo. Ghi rõ phần tự cài, thư viện, mô hình dùng lại và việc sử dụng AI. Xuất PDF theo mẫu `NhomXX_DeTai08.pdf`.
- Hoàn thiện README để tái tạo bảng từ repo sạch, chia sẻ log huấn luyện và quyền truy cập repo với giảng viên. Chuẩn bị slide, demo nhập câu tiếng Anh để xem bản dịch và attention, mỗi thành viên giải thích được cả một phần do bạn khác phụ trách.
