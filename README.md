## Rudra Mantri

**ML/GenAI engineer at a venture-debt firm.** I build production LLM systems by day and hunt
**silent correctness bugs** the rest of the time: no exception, no warning, just numbers that are
quietly wrong.

I care about the parts of AI engineering that don't demo well: evaluation, observability, latency
budgets, and failure modes.

---

### Selected work

| Project | What it is | Evidence |
|---|---|---|
| **[FinGround](https://github.com/RudraMantri123/finground)** | Financial RAG over SEC 10-Ks that cites every claim, refuses when the filings can't support an answer, and benchmarks *hallucination*, not just accuracy | LLM judge agrees with human labels at **κ = 0.84**; retrieval regressions **fail CI** |
| **[ThirdChair](https://github.com/RudraMantri123/thirdchair)** | Real-time voice teammate for meetings: streaming speech-to-text → LLM → text-to-speech, with barge-in | **711 ms p50** end-to-end on-device, with a per-turn latency budget |
| **[mps-sentry](https://github.com/RudraMantri123/mps-sentry)** | Differential checks for PyTorch's Apple Silicon backend: run on MPS and CPU, flag what float rounding can't explain | Both checks reproduce real bugs from the investigations below |
| **Hugging Face diffusers** | Root-caused silent miscompute on Apple Silicon | [#14459](https://github.com/huggingface/diffusers/pull/14459), [#14556](https://github.com/huggingface/diffusers/pull/14556), [#14620](https://github.com/huggingface/diffusers/pull/14620) (in review) |

---

### FinGround: evaluation is the product

- **Hybrid retrieval** (BM25 + dense, RRF) with cross-encoder reranking, company-aware filtering, and
  table-aware 10-K ingestion that keeps financial statements as tables.
- **Cite-or-refuse answering:** every claim carries a `[C#]` citation back to a filing section, and an
  evidence gate abstains before a token is spent when retrieval is weak.
- **Four-metric eval harness:** correctness, citation accuracy, hallucination rate and refusal
  calibration, with bootstrap confidence intervals and per-question audit dumps.
- **Production layer:** request tracing (retrieved chunks, rerank movement, tokens, cost), p50/p95
  latency, citation coverage, failure rate, and a dashboard to root-cause a bad day.
- **Findings so far:** a silent context-truncation bug cost **8× correctness and 90× hallucination**
  with identical weights; three "finance-tuned" open models **lost to their general base model** on
  every grounding metric.

### ThirdChair: real-time means failure modes

- Every turn is timed from end of speech to first audible audio: STT, LLM time-to-first-token,
  TTS first byte, playback, shown as a stacked breakdown with p50/p95.
- **Nothing hangs:** deadlines on every stage; a slow model says *"One moment."*, a dead cloud
  service falls back to on-device engines, and a timeout apologises and recovers.
- **Replay mode** feeds recorded sessions back through the pipeline as a regression test.
- Runs **fully local** (whisper.cpp + Ollama) when nothing may leave the machine.

<details>
<summary><b>The Apple Silicon investigations</b> (how two silent bugs were found)</summary>

<br>

- **SDXL rendered all-black images** with attention slicing on MPS. Traced through the denoiser to
  `baddbmm(input=empty, beta=0)`: PyTorch documents that `input` is ignored, but the MPS kernel
  propagated NaN from it, so recycled allocator pages poisoned the attention scores
  (see [pytorch#187521](https://github.com/pytorch/pytorch/issues/187521)). The diffusers fix is
  buffer-free and benchmarks ~37% faster than the path it replaces.
- **Model weights silently corrupted** during `device_map="mps"` loading. Reproduced without the
  30 GB model by building a faithful miniature; the cause is non-blocking host→device copies reading
  temporaries that were already freed
  (see [pytorch#189690](https://github.com/pytorch/pytorch/issues/189690)). Version-bisected across
  torch 2.10–2.13.
- Write-up: **[Your Mac is quietly computing the wrong answer](https://github.com/RudraMantri123/mps-sentry/blob/main/docs/your-mac-is-computing-the-wrong-answer.md)**

</details>

---

### How I work

Root-cause over workaround. A bug isn't understood until it's reproduced in the smallest form that
still fails, and every fix ships with a test that provably fails without it. Numbers in my READMEs
are measured, with the setup stated.

**Toolbox:** Python · TypeScript · PyTorch · FastAPI · PostgreSQL · FAISS · sentence-transformers ·
Ollama · WebSockets · GitHub Actions

**Interested in:** evaluation and grounding, inference correctness, retrieval, real-time AI systems,
and where finance and ML actually meet.

[rudramantri.dev@gmail.com](mailto:rudramantri.dev@gmail.com) · Open to on-site roles internationally
