# Kiểm chứng câu trả lời QA bằng NLI

Đồ án môn Chuyên đề Xử lý ngôn ngữ tự nhiên, nhóm 2 người.

**Ý tưởng.** Hệ thống hỏi đáp thường trả lời sai mà vẫn rất tự tin. Nhóm dùng mô hình NLI kiểm tra xem
đoạn văn (premise) có *kéo theo* câu trả lời (hypothesis) hay không, để:

1. **Biết từ chối:** nếu đoạn văn không kéo theo câu trả lời thì không trả lời (selective QA).
2. **Chọn lại câu trả lời** *(phương pháp đề xuất)*: reader trả về top-k ứng viên, NLI chấm từng ứng viên,
   chọn ứng viên có điểm `λ·P_reader + (1−λ)·P_entail` cao nhất.

## Câu hỏi nghiên cứu

- **RQ1:** Điểm NLI có giúp từ chối đúng lúc tốt hơn độ tự tin của reader không? (AURC, AUROC)
- **RQ2:** Kích thước verifier (xsmall, base, large) và cách đưa đầu vào (chia premise, tạo hypothesis)
  ảnh hưởng thế nào?
- **RQ3:** Xếp hạng lại top-k bằng NLI có sửa được câu trả lời sai của reader không? Kết hợp
  reader và NLI (λ, hồi quy logistic) có tốt hơn từng tín hiệu riêng lẻ không?

## Pipeline

```
SQuAD 2.0 dev ──► Reader (roberta-base-squad2) ──► top-k ứng viên + xác suất (gồm null)
                                                     │
                       (q, a) ──► hypothesis: concat | template | llm
                                                     │
          đoạn văn ──► chia: full | sent | win3 ──► NLI verifier ──► max P(entail) theo các mảnh
                                                     │
          B1/B2 (reader) · V (chỉ NLI) · R (xếp hạng lại) · C (hồi quy logistic) ──► ngưỡng τ ──► trả lời / từ chối
```

| Ký hiệu | Phương pháp | Độ tự tin |
|---|---|---|
| B1 | reader-prob | softmax của span tốt nhất (có tính null) |
| B2 | reader-margin | logit span − logit null |
| V | NLI verify | P(entail) của span tốt nhất |
| **R** | **NLI rerank (đề xuất)** | chọn lại trong top-k theo `λ·P_reader + (1−λ)·P_entail` |
| C | combo | hồi quy logistic trên [P_reader, margin, P_entail, P_contra] |
| RC | rerank + combo | chọn câu trả lời như R, độ tự tin bằng hồi quy logistic trên [P_reader, margin, P_entail, P_contra] của ứng viên được chọn, P_entail của top-1 và null prob |

Đánh giá: **AURC** (diện tích dưới đường risk–coverage, thấp hơn là tốt hơn), AUROC, độ chính xác ở mức
coverage 50/80/100%, SQuAD2-F1 khi có ngưỡng từ chối. Một câu trả lời được tính là đúng nếu F1 ≥ 0.5 so
với đáp án. Trả lời một câu không có đáp án luôn bị tính là sai.
Tập dữ liệu được chia cố định bằng hash id: **calib 25%** (để chọn setting, λ, τ) và **test 75%** (chỉ để báo cáo).
Có bootstrap 95% CI cho AURC, và paired bootstrap cho hiệu AURC so với B1/B2 (`significance.md`).

## Cấu trúc

```
qaverify/
  data.py         tải SQuAD 2.0, lấy mẫu, chia calib/test, jsonl
  reader.py       reader trích xuất, trả về top-k ứng viên       (thành viên A)
  hypothesis.py   concat / template / LLM viết lại              (thành viên A)
  verifier.py     NLI verifier, chia premise                     (thành viên B)
  methods.py      B1, B2, V, R, C, RC                                (thành viên B)
  metrics.py      EM/F1, risk-coverage, AURC, AUROC, bootstrap   (cả hai)
scripts/
  run_reader.py   bước 1: tạo runs/<run>/preds.jsonl (--dataset squad_v2_dev | addsent | addsent_orig | mrqa:<Tên>)
  finetune_verifier.py  bước 2b: tinh chỉnh nli-base trên cặp QA→NLI từ SQuAD 2.0 train -> runs/models/nli-base-qa
  run_verify.py   bước 2: tạo runs/<run>/scores/<verifier>.jsonl (mỗi verifier một file, chạy song song được)
  analyze.py      bước 3: gộp kết quả, tạo runs/<run>/results/{main.md, ablation.md, significance.md, all.csv, risk_coverage.png, aurc_vs_size.png, settings.json}
  find_examples.py  bước 4: chọn ví dụ cho slide (sửa đúng, làm sai, bắt được, bỏ sót) -> results/examples.md
  data_analysis.py  bước 5: phân tích bộ dữ liệu (loại câu hỏi, độ dài, câu không đáp án, plausible answer,
                    trần của re-ranking, P_entail theo nhóm) -> results/data_analysis.md + data_*.png
  deep_analysis.py  bước 7: NLI và trùng lặp từ vựng (HANS), loại câu không đáp án trên toàn bộ test, bootstrap theo cụm,
                    nhất quán qua 35 bài, đổi tiêu chí đúng, cỡ calib, bỏ từng đặc trưng của C, ECE -> results/deep_analysis.md
  compare_verifiers.py  bước 8: kiểm định cặp giữa các verifier (vd. bản tinh chỉnh so với bản gốc) và kiểm tra trùng lặp
                    từ vựng với cùng một setting cho mọi verifier -> results/compare_verifiers.md, compare_hans.png
  edge_cases.py     bước 9: 11 edge case của dữ liệu (qaverify/edge.py) và mức ảnh hưởng tới B2, C -> results/edge_cases.md
  error_analysis.py bước 10: phân tích lỗi sau khi có kết quả: phân rã lỗi, edge case trong lỗi còn lại, reader và NLI
                    sai ở câu khác nhau, ví dụ lỗi tự tin nhất -> results/error_analysis.md
  ood_eval.py       bước 11: đánh giá ngoài miền (AddSent, Natural Questions, NewsQA): học τ, λ, C trên calib SQuAD,
                    áp dụng nguyên xi -> <source>/results/ood_eval.md
  risk_control.py   bước 12: chọn ngưỡng có bảo đảm rủi ro ≤ α (Learn-then-Test, kiểm định nhị thức), kiểm tra tính hợp lệ
                    bằng 200 lần chia đôi, và thử bảo đảm đó ngoài miền -> results/risk_control.md
  make_annotation.py     bước 6a: tạo bộ gán nhãn loại câu không đáp án (Rajpurkar 2018): --mode random (sheet_*), check (kappa), errors (câu C trả lời sai)
  analyze_annotation.py  bước 6b: độ đồng thuận (kappa) và hành vi reader/NLI theo từng loại -> annotation/<set>_annotation.md
run_pipeline.ipynb  notebook Colab chạy toàn bộ
tests/            pytest
```

## Chạy trên Colab (GPU T4)

**Cách nhanh nhất:** tải `doan.zip` vào `MyDrive/nlp_doan/`, mở `run_pipeline.ipynb` trên Colab, chọn T4 GPU,
đặt `ROLE` rồi Run all. Mặc định notebook chạy toàn bộ SQuAD 2.0 dev (`N = 11873`, run `squad11k`); đặt `N = 2000` để chạy bản thử.
Chạy song song hai tài khoản: thư mục `nlp_doan/` chia sẻ quyền Editor và được thêm shortcut vào My Drive của người kia;
người A đặt `ROLE = "A"` (reader, nli-xs, nli-base), người B đặt `ROLE = "B"` (đợi `preds.jsonl`, chạy nli-large).
Phần tổng hợp chỉ chạy khi đủ cả 3 verifier. Kết quả được lưu trong `MyDrive/nlp_doan/runs/`. Nếu Colab bị ngắt thì Run all lại,
các bước đã xong sẽ tự bỏ qua.

Nếu muốn chạy bằng lệnh:

```bash
!git clone <repo> doan && cd doan && pip install -q -r requirements.txt
!python -m pytest -q tests
!python scripts/run_reader.py --n 2000 --out runs/squad2k            # ~2 phút
!python scripts/run_verify.py --run runs/squad2k --verifier nli-xs   # mỗi verifier chạy riêng
!python scripts/run_verify.py --run runs/squad2k --verifier nli-base
!python scripts/run_verify.py --run runs/squad2k --verifier nli-large
!python scripts/analyze.py --run runs/squad2k
```

Muốn có thêm hypothesis viết lại bằng LLM thì thêm `--llm Qwen/Qwen2.5-1.5B-Instruct` vào `run_reader.py`
(bước verify khi đó dùng `--hyps concat,template,llm`).
Nên lưu thư mục `runs/` lên Google Drive chung để cả hai thành viên dùng chung kết quả.

Chạy thử nhanh trên CPU: `--n 200` ở bước 1, `--limit 50` ở bước 2.

## Phân công

| | A: phía QA | B: phía NLI |
|---|---|---|
| Code | `reader.py`, `hypothesis.py` (+ LLM), sau seminar thêm retriever cho RAG | `verifier.py`, `methods.py`, sau seminar tinh chỉnh QA→NLI |
| Câu hỏi nghiên cứu | RQ1, phần RAG | RQ2, RQ3 |
| Cùng làm | `metrics.py`, `analyze.py`, phân tích lỗi, slide, báo cáo | |

Giao ước dữ liệu: `preds.jsonl` (do A tạo) có `id, question, context, answers, split, candidates[{text, prob, logit}],
null_logit, null_prob, hyp{concat, template, llm?}`. B đọc file này và tạo `scores/<verifier>.jsonl` gồm `{id, nli: {"verifier|hyp|chunk": [[ent, con] cho từng ứng viên]}}`.

## Lịch (seminar 7/11, nếu dời sang 14/11 thì có thêm 1 tuần)

| Ngày | A | B | Cùng làm |
|---|---|---|---|
| 26/10 | | | Họp phân vai, tạo repo và Drive |
| 27–28/10 | Chạy `run_reader` 2000 câu, đọc Chen 2021, Kamath 2020 | Chạy `run_verify` nli-xs, nli-base, đọc SummaC, AlignScore | |
| 29–30/10 | Thêm hypothesis bằng LLM | Chạy nli-large | Chạy `analyze` lần đầu |
| 31/10–1/11 | Xem lỗi template và LLM | Xem λ, combo | **Chốt số liệu seminar** |
| 2–3/11 | Slide: bài toán, pipeline | Slide: công trình liên quan, kết quả | Chọn 3 ví dụ lỗi |
| 4–6/11 | | | Tập 2 lần, chuẩn bị câu hỏi |
| **7/11** | | | **Seminar** |
| Sau seminar | RAG trên NQ-open (BM25 + dense, LLM reader) | Tinh chỉnh verifier trên dữ liệu QA→NLI từ SQuAD train | Phân tích lỗi, demo, báo cáo |

## Tài liệu tham khảo chính

- Chen, Choi, Durrett (2021). *Can NLI Models Verify QA Systems' Predictions?* Findings of EMNLP.
- Kamath, Jia, Liang (2020). *Selective Question Answering under Domain Shift.* ACL.
- Laban et al. (2022). *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL.
- Zha et al. (2023). *AlignScore: Evaluating Factual Consistency with a Unified Alignment Function.* ACL.
- Rajpurkar, Jia, Liang (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL.
- Geifman, El-Yaniv (2017). *Selective Classification for Deep Neural Networks.* NeurIPS.
