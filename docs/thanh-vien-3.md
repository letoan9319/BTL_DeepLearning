# Thành viên 3 · Transformer và EnViT5

## Phương pháp

**Transformer tự huấn luyện.** Mỗi câu tiếng Anh được thêm BOS và EOS, sau đó đi qua embedding 256 chiều, positional encoding hình sin và ba tầng encoder. Mỗi tầng có bốn head attention, mạng feed-forward 256–1.024–256 dùng ReLU, kết nối residual, dropout 0,1 và layer normalization. Decoder có ba tầng với masked self-attention và cross-attention lên encoder. Trong mỗi head, mô hình chia tích vô hướng của query và key cho căn bậc hai của chiều head (64), che các vị trí không hợp lệ trước softmax và dùng trọng số thu được để lấy tổng có trọng số của các value. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Khi huấn luyện, decoder nhận tiền tố tham chiếu và dự đoán token kế tiếp. Cross-entropy bỏ qua PAD. AdamW dùng learning rate 0,0003 và gradient clipping ở chuẩn 1. Ba lượt dùng cùng danh sách 133.166 cặp train, seed 42, batch 32 và ngân sách baseline một epoch. Mỗi phía có vocab riêng tối đa 8.000 mục, học trên train. Giới hạn đầu vào và nhãn là 128 token, tính cả BOS và EOS. Số câu bị cắt theo tokenizer được ghi vào cấu hình từng lượt.

**Giải mã.** Greedy chọn token có log-probability cao nhất. Beam giữ tối đa bốn tiền tố đang mở cho mỗi câu và lưu giả thuyết hoàn chỉnh tốt nhất theo tổng log-probability chia số token đã sinh, tính cả EOS. Decoder dừng khi giả thuyết đã hoàn chỉnh thắng cận trên của các tiền tố còn mở hoặc khi đạt 128 bước. Khi chưa sinh EOS ở giới hạn, lấy tiền tố đang mở tốt nhất. Hai decoder chặn PAD và BOS sau bước khởi đầu. Việc lưu key và value của từng tầng cho phép giải mã tăng dần theo từng token.

**EnViT5.** Fine-tune checkpoint dịch Anh–Việt của VietAI với tokenizer gốc, thêm tiền tố ngôn ngữ vào nguồn và đích. Chạy ba epoch trên Colab, learning rate 0,00003, batch hiệu dụng 32 câu trên một GPU, tối đa 128 token mỗi phía. Chọn checkpoint theo loss của tst2012, nạp lại checkpoint đã chọn và sinh bản dịch bằng greedy và beam rộng 4. Bỏ tiền tố `vi:` ở đầu bản dịch trước khi tính điểm. Khi so sánh với Transformer tự huấn luyện cần tính đến dữ liệu song ngữ mà VietAI đã dùng trước bước fine-tune. [Model card EnViT5](https://huggingface.co/VietAI/envit5-translation) · [Ngo và cộng sự, MTet](https://arxiv.org/abs/2210.05610)

## Đọc kết quả

Mỗi tệp dự đoán giữ ID dòng gốc theo thứ tự của split. Notebook 05 chấm corpus BLEU phân biệt hoa thường với tokenizer 13a. Để đo ảnh hưởng của beam, lấy BLEU beam trừ BLEU greedy của cùng checkpoint. Để đo ảnh hưởng của tokenizer, cố định kiến trúc, ngân sách và decoder.

Các hình cross-attention lấy head 1 ở tầng decoder 3, cột là token nguồn và hàng là token đích được sinh. Ba ví dụ validation được chọn trước khi chấm gồm câu dài có số `tst2012-1`, câu phủ định `tst2012-2` và câu có địa danh Saigon `tst2012-7`. Tệp JSON kèm hình chứa câu nguồn, tham chiếu, bản dịch, nhãn hai trục và ma trận để nhóm đối chiếu trực tiếp.
