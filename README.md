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
| [03 · Transformer](notebooks/03_transformer.ipynb) | Cấu hình và kiểm tra causal/padding mask |
| [04 · Fine-tune EnViT5](notebooks/04_finetune_envit5.ipynb) | Đặt `RUN_TRAIN=True` để tải checkpoint và huấn luyện |
| [05 · Đánh giá](notebooks/05_danh_gia.ipynb) | Tính corpus BLEU từ JSONL đã dịch và vẽ attention |

Đây là khung khởi đầu. Hai kiến trúc tự cài, vòng huấn luyện của chúng và phần xuất dự đoán greedy/beam là công việc nhóm triển khai theo kế hoạch. Hiện chưa có checkpoint do nhóm huấn luyện hoặc bảng điểm thực nghiệm.
