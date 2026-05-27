# DESA — Project Explanation and Output Interpretation

## 1. The big picture

**DESA = Dynamic Emotional-State Architecture** — a class project for *Quantitative Methods for NLP* (MIT, Spring 2026). The high-level idea is: instead of one monolithic empathetic chatbot, you train **four small "expert" adapters**, each specialized to a quadrant of emotional space, and then **dynamically mix them** based on the conversational context.

The base model is `mistralai/Mistral-7B-Instruct-v0.2`. The dataset is **EmpatheticDialogues** (the `Estwld/empathetic_dialogues_llm` mirror), which has ~32 fine-grained emotion labels per dialogue.

The pipeline is four notebooks (`01_clustering` → `02_train_adapters` → `03_train_gating` → `04_evaluation`), and four "systems" are evaluated against each other.

---

## 2. The clustering — VAD quadrants (Notebook 01)

Each fine-grained emotion (32 of them: `joyful`, `terrified`, `sad`, `excited`, …) is mapped to a point in **NRC-VAD** space: a triple $(v, a, d) \in [0,1]^3$ for **valence, arousal, dominance**. DESA throws away dominance and clusters using only $(v, a)$ via a **deterministic threshold at 0.5**:

$$
\text{cluster}(v, a) = \begin{cases}
0 & v \ge 0.5,\ a \ge 0.5 \quad \text{(positive, high-arousal: excited, joyful)} \\
1 & v \ge 0.5,\ a < 0.5 \quad \text{(positive, low-arousal: content, grateful)} \\
2 & v < 0.5,\ a \ge 0.5 \quad \text{(negative, high-arousal: angry, terrified)} \\
3 & v < 0.5,\ a < 0.5 \quad \text{(negative, low-arousal: sad, lonely)}
\end{cases}
$$

The proposal had originally suggested K-means on VAD vectors, but `clustering.py:79-87` (`emotion_to_quadrant`) replaces that with the deterministic rule. This guarantees stable, semantically named clusters across machines and across re-runs (no random init drift).

The training set is then partitioned by `assignments[emotion]`, producing four jsonl files `cluster_{0..3}_train.jsonl` with a 90/10 train/val split.

---

## 3. The four LoRA experts — QLoRA fine-tuning (Notebook 02)

For each cluster $k \in \{0,1,2,3\}$, you train a **low-rank adapter** on top of frozen, 4-bit-quantized Mistral.

### LoRA math

For each target attention projection $W \in \mathbb{R}^{d \times d}$ (specifically `q_proj`, `k_proj`, `v_proj`, `o_proj`), LoRA replaces the forward pass:

$$
y = W x \;\;\longrightarrow\;\; y = W x + \frac{\alpha}{r} \, B A x,
$$

where $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$ are trainable, and $\Delta W = BA$ is a rank-$r$ update. From `configs/qlora_config.yaml` you have $r = 16$, $\alpha = 32$, so the scale factor is $\alpha/r = 2$. Dropout 0.05 on the LoRA path, no bias terms.

### QLoRA twist

The base $W$ is frozen and stored in **NF4** (a 4-bit normal-float quantization), with `bnb_4bit_compute_dtype: float16`. Dequantization happens just-in-time for the matmul. The trainable LoRA params are deliberately cast back to **fp32** in `train_adapter.py:150` (`cast_trainable_parameters_to_fp32`) so AMP gradient scaling works cleanly — a known gotcha when mixing bitsandbytes with HF Trainer's BF16/FP16 paths.

### Loss

Standard causal-LM next-token cross-entropy on chat-templated turns. Each conversation produces one SFT example per assistant utterance via `conversation_to_sft_texts`. Training: 3 epochs, lr $2 \times 10^{-4}$, cosine schedule, effective batch size 16 (= 2 × 8 grad-accum), cap of 4000 examples per cluster, max_seq_length 512.

The result of step 02 is four directories `outputs/adapter_cluster_{k}/final/` containing $\{A^{(k)}, B^{(k)}\}$ for each of the four target modules at each transformer layer.

---

## 4. The gating heads (Notebook 03)

Now you have four experts. The interesting question: at inference time, **how do you decide how to combine them**? The project trains two gating mechanisms.

### 4.1 Turn-level gate (`gating.py:19-47`)

This is one shared gate that, given the conversation context $x$, outputs a single 4-vector of mixing weights for the whole reply. It's trained as a 4-class classifier targeting the gold cluster.

Let $h \in \mathbb{R}^{T \times d}$ be the **last hidden state** of the frozen base model (Mistral, $d = 4096$) over the prompt. Mean-pool with the attention mask:

$$
\bar h = \frac{\sum_{t=1}^T m_t \, h_t}{\sum_{t=1}^T m_t}, \qquad m_t \in \{0, 1\}.
$$

Then a small MLP head:

$$
g(\bar h) = \mathrm{softmax}\!\left( W_2 \cdot \mathrm{ReLU}(W_1 \bar h) \right) \in \Delta^3,
$$

with $W_1 \in \mathbb{R}^{1024 \times 4096}$, $W_2 \in \mathbb{R}^{4 \times 1024}$, dropout 0.1. (The bottleneck dim is `hidden_dim // 4` = 1024.)

Trained with cross-entropy against the gold cluster $y \in \{0,1,2,3\}$ derived from `cluster_assignments`:

$$
\mathcal{L}_{\text{turn}} = - \log g(\bar h)_{y}.
$$

2 epochs, AdamW, lr $10^{-4}$, balanced sampling of 512 examples per cluster.

**Then at inference** (`inference.py:251-276`), the predicted weights $\alpha = g(\bar h)$ are pushed into PEFT's weighted-adapter API:

$$
\Delta W_{\text{blend}} = \sum_{k=0}^{3} \alpha_k \, \Delta W^{(k)} = \sum_k \alpha_k \, B^{(k)} A^{(k)}.
$$

So the LoRA delta applied during generation is a convex combination of the four cluster experts.

### 4.2 X-LoRA — token- and layer-level gating (Notebook 03 cont., `inference.py:90-129`)

This is the more ambitious "Mixture of LoRA Experts" approach using PEFT's native `XLoraConfig`. Instead of one global $\alpha$, you get a scalar **per token, per layer, per adapter**:

$$
\Delta W^{(\ell)}_{t} = \sum_{k=0}^{3} g^{(\ell)}_{t,k} \, B^{(k,\ell)} A^{(k,\ell)},
$$

where $g^{(\ell)}_{t,k}$ comes from a small classifier on top of frozen Mistral with `xlora_depth=4`, `xlora_size=2048`, dropout 0.1, `enable_softmax=True`, `layerwise_scalings=True`. The classifier produces a tensor of shape `(batch, seq_len, n_layers, n_adapters)`.

A subtle constraint enforced in `inference.py:97-101`: PEFT's X-LoRA path is incompatible with bitsandbytes 4-bit linears, so for system 4 the base must be loaded **unquantized in FP16**.

### 4.3 The VAD stretch goal (`gating.py:50-65`)

Not used in evaluation. It would route based on a tracked VAD state with an EMA "emotional inertia": $\mathbf{v}_t = (1-\mu) \mathbf{v}_t^{\text{cur}} + \mu \mathbf{v}_{t-1}$, comparing to learned per-adapter prototypes via cosine similarity, then softmax. Stretch goal — not wired into inference.

---

## 5. The four systems being compared (Notebook 04)

1. **`static_prompt`** — vanilla Mistral with a hand-written system message that names the gold emotion; **all adapters disabled**. Baseline.
2. **`argmax_adapter`** — set adapter to `cluster_{assignments[gold_emotion]}`. Cheats by using the gold label, no learned routing.
3. **`turn_level`** — turn gate predicts $\alpha$, weighted blend applied via `add_weighted_adapter` / `set_adapters`, single $\alpha$ for the whole reply.
4. **`token_level_xlora`** — full X-LoRA, one mixture per token per layer.

---

## 6. The metrics and their math

- **Gold-response perplexity** (`evaluate.py:111-149`): for each test example, build prompt = full history minus last assistant turn, response = last assistant turn. Mask prompt tokens with `-100` so they're excluded from the loss, encode `[prompt | response | eos]`, and compute

$$
\mathrm{PPL} = \exp\!\left(\frac{1}{N}\sum_{i=1}^{N} -\log p_\theta(y_i \mid y_{<i}, x)\right).
$$

Lower = the model finds the gold response less surprising. The token count $N$ is summed across the whole test slice — this is a corpus-level PPL, not a per-example mean.

- **Distinct-1 / Distinct-2** (lexical diversity):

$$
\text{Distinct-}n = \frac{|\{\text{unique } n\text{-grams in all generations}\}|}{|\{\text{total } n\text{-grams in all generations}\}|}.
$$

Higher = less repetitive.

- **Emotion accuracy**: pipe each generation into `j-hartmann/emotion-english-distilroberta-base` (a 7-way classifier: anger, disgust, fear, joy, sadness, surprise, neutral). Map the gold ED-32 label down to the same 7-way space via `EMOTION_TO_BROAD`, then count exact matches.

- **Gating entropy**: $H(\alpha) = -\sum_k \alpha_k \log \alpha_k$, max $= \log 4 \approx 1.3863$. Reports mean and std across the test set.

- **Gating alignment** (`gating_alignment`): the average mass placed on the gold cluster, $\overline{\alpha_{y}} = \frac{1}{N}\sum_i \alpha^{(i)}_{y_i}$. Uniform random would give $0.25$.

Per-cluster alpha breakdowns (`cluster_X_mean_alpha_Y`) condition on gold class $X$ and report mean weight on adapter $Y$. A perfect gate would be a $4 \times 4$ identity: row $X$ peaked on column $X$.

---

## 7. Reading the actual results

```
                  PPL ↓     Dist-1   Dist-2   EmoAcc
static_prompt     22865.9   0.127    0.416    0.760
argmax_adapter       39.1   0.277    0.701    0.295
turn_level           30.7   0.281    0.660    0.225
token_level_xlora  7272.5   0.185    0.477    0.170
```

This table is full of red flags. Let me work through them.

### 7.1 Why is `static_prompt` PPL = 22,866?

That's astronomical — it implies the model assigns near-zero probability to gold tokens. The cause is almost certainly a **prompt-format mismatch**: `static_prompt`'s **generation** prompt embeds the conversation inside an instruction-style "You are an empathetic assistant…" wrapper (`inference.py:177-187`), but its **perplexity** is computed via the default `_prompt_and_gold` path in `evaluate.py:83-86`, which just chat-templates the history. So the PPL number for `static_prompt` is *not* measuring the static system — it's measuring whatever adapter happened to be active on the multi-adapter PEFT model when this row was evaluated, against a plain prompt. The huge value is consistent with an adapter being mid-blended or mis-set rather than disabled. If you want a fair PPL, `evaluate_system` for `static_prompt` should be invoked with `prompt_builder=build_static_emotion_prompt` and `context_manager=adapters_disabled`.

(`static_prompt` does win on emotion accuracy — 0.76 vs ~0.2 for the others. That's because with an explicit "respond with emotion = {gold}" instruction, the off-the-shelf classifier easily picks up the requested tone. But this is a near-cheating signal, not capability.)

### 7.2 The cluster-adapter systems crush PPL (39 and 31)

Both `argmax_adapter` and `turn_level` get PPL in the 30–40 range — three orders of magnitude better than the static baseline as measured. Two interpretations:

- **Real**: the adapters genuinely fit the EmpatheticDialogues distribution, so the gold response is much less surprising once you condition on a cluster expert.
- **Format effect**: if the static row was inflated by a prompt mismatch, the "fair" gap would be smaller. But even after fixing that, you'd expect the QLoRA-finetuned models to win on PPL — that's exactly what fine-tuning on the train distribution does.

Distinct-1/2 also doubles for the adapter systems (~0.28 / ~0.66 vs 0.13 / 0.42). The base model is more repetitive in this scoring; the fine-tunes produce more lexical variety, possibly because QLoRA at lr 2e-4 nudges the model toward the somewhat varied EmpatheticDialogues style.

Lower emotion accuracy for the adapter systems (0.23–0.30) is the interesting failure mode: **PPL good, vibe-matching worse**. Without an explicit emotion prompt, the response style isn't always pegged to the gold quadrant. This is exactly what DESA's gating is supposed to fix.

### 7.3 The turn-level gate has collapsed

Look at the per-cluster mean-alpha blocks. All four rows are nearly identical:

$$
\bar{\alpha}\,\big|\,\text{cluster } 0..3 \;\approx\; (0.306,\ 0.356,\ 0.062,\ 0.277).
$$

The gate is producing the **same vector regardless of input**. Two pieces of corroborating evidence:

- `std_entropy ≈ 4.4 \times 10^{-16}` — that's machine epsilon. Entropy is **constant** across all examples.
- `mean_entropy = 1.257` is below $\log 4 = 1.386$, consistent with the constant vector $(0.306, 0.356, 0.062, 0.277)$:

$$
H = -\sum_k \alpha_k \log \alpha_k \approx -(0.306 \log 0.306 + 0.356 \log 0.356 + 0.062 \log 0.062 + 0.277 \log 0.277) \approx 1.257.
$$

Yes — the math confirms the gate is outputting that constant vector for every prompt. It learned a **prior** over clusters, not a context-conditional posterior.

Why? Likely culprits:

1. The mean-pool over a long prompt washes out the emotional signal — the classifier ends up on a near-constant pooled feature.
2. Only 2 epochs at lr $10^{-4}$ over 4 × 512 = 2048 examples may not be enough for a head sitting on top of a frozen 4096-dim feature.
3. The MLP collapsed because input variation is small relative to the bias terms.

`gold_cluster_mass = 0.227`, **below** uniform 0.25. With a constant $\alpha$, the expected gold mass is $\sum_c P(\text{gold}=c) \cdot \alpha_c$. The gate assigns only 0.062 to cluster 2 (`negative_high_arousal`), and any test example from that cluster drags the average below 0.25.

**Bottom line:** the turn-level system gets its low PPL essentially by accident — it's doing roughly a fixed convex combination of all four adapters that happens to hold up. It is **not routing**.

### 7.4 X-LoRA is also collapsed — toward uniform

```
mean_entropy   = 1.3862917
max_entropy    = 1.3862944   (= log 4)
std_entropy    = 4.7e-6
```

This is **maximum entropy at machine precision**: the X-LoRA classifier is outputting $(\tfrac14, \tfrac14, \tfrac14, \tfrac14)$ per token, per layer. The per-cluster alphas confirm it — every entry is 0.249–0.250.

That implies the X-LoRA classifier head was either (a) never actually trained — `xlora_classifier.pt` may just be an init checkpoint — or (b) trained but with the softmax saturated to the uniform fixed point. Given how X-LoRA works (it sums **all four** adapter deltas at every layer at every token), a uniform gate is essentially the worst case: you're applying

$$
\Delta W^{(\ell)}_{\text{xlora-uniform}} = \frac{1}{4} \sum_{k=0}^{3} B^{(k,\ell)} A^{(k,\ell)},
$$

which is the average of four adapters that were each trained to specialize in a different emotion. The result is a low-rank update that doesn't represent any single quadrant — predictably destroying perplexity. PPL = 7,272 is exactly what you'd expect.

### 7.5 What the numbers tell the proposal

- **Adapters work.** The QLoRA fine-tunes individually fit EmpatheticDialogues well (PPL 39 with the gold-cluster expert).
- **Routing does not yet work.** Both gating mechanisms degenerated:
  - The turn gate collapsed to a fixed prior — mean-pool + small-data CE didn't produce input dependence.
  - X-LoRA collapsed to uniform — its classifier likely wasn't trained to convergence (or wasn't loaded), and X-LoRA at uniform weights is worse than picking any single adapter.
- **The adapter systems lose on emotion accuracy** because the final reply doesn't carry an explicit emotion cue. With routing fixed, this is the metric that should rise toward `static_prompt`'s 0.76 while keeping PPL low.

---

## 8. Concrete fixes the numbers point to

If you want to make the comparison fair and the gating actually do something:

1. **Re-run `static_prompt` PPL** with the same prompt builder used at generation (`build_static_emotion_prompt`) and inside `adapters_disabled`. The current 22,866 is misleading.
2. **Diagnose the turn gate.** Check whether `turn_gate.pt` predictions vary across batches — print $\alpha$ for ten different prompts. If they're identical, retrain with: more epochs (≥10), use the gold emotion's pooled hidden state rather than mean-pool over the whole prompt, or use a class-balanced loss with label smoothing.
3. **X-LoRA classifier** — verify it actually trained (loss curve in notebook 03), and that `xlora_classifier.pt` contains a non-trivial state dict (not just init). If the classifier state never made it in, `load_state_dict(strict=False)` would silently leave it at random init, then the softmax would converge to uniform.
4. The X-LoRA path is also incompatible with 4-bit base (enforced at `inference.py:97-101`), which means it ran in FP16 — confirm this in the notebook to rule out memory-related truncation.
