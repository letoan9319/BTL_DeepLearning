# Dịch máy Anh–Việt trên IWSLT 2015

Bài tập lớn môn Học sâu, chủ đề 8. Nhóm so sánh hai mô hình tự cài đặt (LSTM với attention và Transformer nhỏ), dùng EnViT5 fine-tune làm mốc tham chiếu. Xem [kế hoạch và phân công](docs/ke-hoach.md).

## Chạy trên máy

Từ thư mục gốc của repo, dùng Python 3.10 trở lên. Ví dụ với PowerShell trên Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\jupyter-lab.exe notebooks
```

Riêng lệnh `python scripts/download_data.py` chỉ cần thư viện chuẩn. Script tải sáu tệp UTF-8 vào `data/raw/iwslt15/` và kiểm tra SHA-256 mỗi lần chạy. Bộ dữ liệu do Stanford NLP công bố; script dùng bản sao cố định có cùng checksum. Dữ liệu thô và thư mục kết quả `artifacts/` được giữ trên máy. [Nguồn Stanford](https://nlp.stanford.edu/projects/nmt/) · [Bản sao](https://github.com/stefan-it/nmt-en-vi/tree/93193ddf269449c98495e3455b2bb261b4851234/data)

| Tập | Tệp Anh/Việt | Cặp gốc | Cặp dùng sau lọc dòng trống |
|---|---|---:|---:|
| Train | train.en / train.vi | 133.317 | 133.166 |
| Validation | tst2012.en / tst2012.vi | 1.553 | 1.553 |
| Test | tst2013.en / tst2013.vi | 1.268 | 1.268 |

## Notebook

| Notebook | Điểm bắt đầu |
|---|---|
| [01 · Dữ liệu và tokenizer](notebooks/01_du_lieu_tokenizer.ipynb) | Đếm các split, hàm học BPE/Unigram và tách từ tiếng Việt |
| [02 · LSTM attention](notebooks/02_lstm_attention.ipynb) | Luong attention và kiểm tra mask nguồn |
| [03 · Transformer](notebooks/03_transformer.ipynb) | Kiểm tra mô hình, train trên máy, xuất dự đoán và thử dịch từ checkpoint |
| [04 · Fine-tune EnViT5](notebooks/04_finetune_envit5.ipynb) | Mở trên Colab GPU, chạy từ đầu và lưu checkpoint cùng bản dịch trên Drive |
| [04b · EnViT5 LoRA r=32](notebooks/04b_envit5_lora_r32.ipynb) | Chạy trên Colab GPU, lưu adapter vào `artifacts/envit5-lora-r32/best/` và xuất JSONL cho notebook 05 |
| [05 · Đánh giá](notebooks/05_danh_gia.ipynb) | Tính corpus BLEU từ JSONL đã dịch và vẽ attention |

## Tái tạo phần thành viên 3

Phương pháp, quy tắc giải mã và cách đọc hình attention nằm trong [phần báo cáo của thành viên 3](docs/thanh-vien-3.md).

**Transformer trên Windows.** Trong môi trường đã cài requirements, kiểm tra PyTorch nhận GPU bằng lệnh dưới đây. Trên máy RTX 3060 của nhóm dùng Python 3.12 và bản PyTorch CUDA 12.8.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
.\.venv\Scripts\python.exe -m scripts.check_transformer
foreach ($kind in 'bpe','unigram','word') {
    .\.venv\Scripts\python.exe -m scripts.train_transformer --kind $kind --epochs 1 --batch 32
    if ($LASTEXITCODE -ne 0) { break }
}
```

Mỗi lượt học vocab riêng cho hai ngôn ngữ trên train, lưu tokenizer, cấu hình, log TensorBoard, loss và checkpoint có validation loss thấp nhất trong `artifacts/transformer/<kind>/`. Dùng `--train-only` để chỉ train hoặc `--export-only` để nạp lại checkpoint và xuất bản dịch. Tham số `--output` chọn thư mục riêng cho thí nghiệm mới. Hình và ma trận attention dùng ba câu validation `tst2012-1` (câu dài, số), `tst2012-2` (phủ định) và `tst2012-7` (địa danh), tầng decoder cuối và head 1. Mỗi hàng biểu diễn token đích được dự đoán tại bước tương ứng.

**EnViT5 trên Colab.** Tải notebook 04 lên Colab, chọn GPU và chạy tất cả các ô. Notebook tự tải trực tiếp ba tệp nén IWSLT’15 English–Vietnamese từ bản sao cố định, kiểm tra SHA-256 của sáu tệp ngôn ngữ, tải checkpoint VietAI tại revision ghi trong notebook, fine-tune và xuất kết quả vào `MyDrive/BTL_DeepLearning/artifacts/`. Sao chép hai tệp `predictions/test/envit5-*.jsonl` từ Drive về `artifacts/predictions/test/` của repo. Chạy notebook 05 để tính lại BLEU cho cả Transformer và EnViT5. Bảng so sánh decoder dùng hai bản dịch sinh từ cùng checkpoint và cùng ID test.
