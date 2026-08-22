## Rudra Mantri

ML/GenAI engineer at a venture-debt firm — I build production LLM systems by day, and I hunt
**silent correctness bugs in Apple Silicon ML** the rest of the time.

Silent bugs are the interesting ones: no exception, no warning, no NaN in your inputs, just numbers
that are quietly wrong. They get blamed on the model. Usually they are not the model.

### Current work

**[mps-sentry](https://github.com/RudraMantri123/mps-sentry)** — differential correctness checks for
PyTorch's MPS backend. Runs each operation on MPS and against a CPU reference, and reports only the
disagreement that float rounding cannot explain. Both headline checks came out of real debugging
sessions, below.

**Open-source investigations** (🤗 diffusers):

- **[#14459](https://github.com/huggingface/diffusers/pull/14459)** — SDXL rendered all-black images
  on Apple Silicon with attention slicing. Traced through the denoiser to
  `baddbmm(input=empty, beta=0)`, where PyTorch documents that `input` is ignored — but the MPS
  kernel propagates NaN from it, so recycled allocator pages poison the attention scores
  ([pytorch#187521](https://github.com/pytorch/pytorch/issues/187521)). The fix is buffer-free and
  benchmarks ~37% faster than the path it replaces.
- **[#14556](https://github.com/huggingface/diffusers/pull/14556)** — model weights silently
  corrupting during `device_map="mps"` loading. Reproduced without downloading the 30 GB model in
  the report, by building a faithful miniature; the cause is non-blocking host→device copies reading
  cast temporaries that were already freed
  ([pytorch#189690](https://github.com/pytorch/pytorch/issues/189690)). Version-bisected across
  torch 2.10–2.13.

**[FinGround](https://github.com/RudraMantri123)** — faithfulness-first financial RAG over SEC
filings: inline-citation grounding, abstention when the filings cannot support an answer, and an
LLM-as-judge eval decomposing answers into atomic claims (Cohen's κ = 0.84 against human labels).

Earlier: reproduced published IR baselines end to end (MS MARCO BM25, BGE dense retrieval) and
contributed verified reproduction logs to
[Anserini](https://github.com/castorini/anserini/pull/3038) and
[Pyserini](https://github.com/castorini/pyserini/pull/2352).

### How I work

Root-cause over workaround; a bug is not understood until it is reproduced in the smallest form that
still fails. Every fix ships with a test that provably fails without it — I check both directions,
because a regression test that never fails is decoration.

### Interests

Inference correctness, model loading and memory, quantization, retrieval evaluation, and the places
where finance and ML actually meet.

Open to on-site roles and collaborations internationally.
