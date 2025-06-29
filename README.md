## 🧠 Deterministic Expert Routing via Token Markers

### 🧠 Deterministic Expert Routing via Token Markers

The BearMuzzle sidekick model can be structured as a Mixture of Experts (MoE) model where each expert specializes in a specific task, such as reasoning, linting, planning, or neutral token handling. Instead of relying on a learned gating mechanism, expert routing can be **deterministically controlled** based on token markers emitted by the main LLM.

For example:
- `[THINK_START]` can activate the "Reasoning" expert.
- `[COMMENT]` can activate a "Linter" expert.
- `[PLAN_BEGIN]` can activate a "Planner" expert.

This deterministic approach eliminates the overhead of a learned gating network, allowing consistent, reproducible expert routing at runtime with minimal latency.

#### 🛠 Implementation Strategy in `llama.cpp`

The `llama.cpp` engine supports LMoE (Mixture of Experts) models with runtime expert routing. By modifying the expert gating logic within the inference loop (e.g., inside `llama_sample_softmax()` or related gating paths), you can inject routing logic like:

```cpp
if (current_mode == REASONING) {
    gate[expert_reasoning] = 1.0f;
    gate[others] = 0.0f;
}
```

The `current_mode` can be determined by parsing the token stream from the main LLM during inference.

#### ✅ Advantages
- No latency from model swapping
- Experts remain in memory (no reloads)
- Clear symbolic alignment with LLM output
- Easily extensible by adding new expert modules

This technique makes it possible to build real-time, structure-aware sidekick models that operate predictably alongside the main LLM during streaming inference.



A lightweight CPU-side model that dynamically adjusts the logits of a larger LLM during inference — for real-time steering, behavioral shaping, or stylistic control without modifying the base model.
---

## 🧰 Sidekick Runtime Implementation

The sidekick model is designed to run in real time, in parallel with the main LLM's inference loop. It processes the same input context and outputs a logit adjustment vector or constrained token sequence at each inference step.

To ensure responsiveness and low resource usage, the sidekick model is optimized for:

- **CPU execution**
- **Low memory footprint (<100MB)**
- **1.58-bit (ternary) model weights** using values of {-1, 0, +1}

This allows the sidekick to keep pace with the GPU-based main LLM, even on commodity hardware such as laptops or embedded servers.

### 🧪 Preferred Runtime: `llama.cpp`

We currently recommend using `llama.cpp` as the execution backend for both the sidekick and the main LLM:

- ✅ Supports concurrent contexts (multi-threaded token-by-token inference)
- ✅ Shared tokenizer and GGUF format for main and sidekick models
- ✅ Built-in support for logit manipulation per token step
- ✅ Extremely efficient on CPUs

The sidekick model can be quantized into `Q2_K` or a ternary-simulating GGUF format and run in a dedicated CPU thread using `llama.cpp`.

### 🧪 Other Runtime Options

| Runtime         | Suitability | Notes |
|------------------|------------|-------|
| **ONNX Runtime** | 🟡 Possible | Custom ops required for ternary inference |
| **tinygrad**     | 🔸 Experimental | Easily modifiable but slower |
| **MLC-LLM**       | 🔸 Possible | TVM backend with future support potential |

### 🧠 Why This Matters

Running the sidekick on CPU ensures:

- Full decoupling from GPU usage
- Predictable timing and logit shaping
- Real-time shaping of the next token's logit distribution before sampling

The runtime choice is crucial to preserve responsiveness and portability, enabling the sidekick to function as a consistent inference-time controller regardless of where the main model is hosted.


### 🧩 Concepts Used

BearMuzzle modulates the behavior of a large language model (LLM) by manipulating the **logits** — the raw scores the LLM assigns to each possible next token before sampling.

#### What Is a Logit?

A *logit* is a raw, unnormalized score representing the model’s preference for a specific token. At each step of generation:

- The model computes a logit for every token in its vocabulary.
- These logits are passed through a softmax function to become probabilities.
- The higher the logit, the more likely that token will be selected.

#### What Is a Logit Penalty?

BearMuzzle applies a **logit penalty vector** to steer the model:

- This vector has one value per token in the vocabulary.
- Each value is a non-negative number subtracted from the corresponding token's logit.
- The higher the penalty, the more that token is discouraged — but not forbidden.

This technique allows nuanced control:
- A penalty of `0.0` leaves the token untouched.
- A penalty of `0.2` subtly reduces the token’s chance.
- A penalty of `5.0` heavily discourages the token.
- Unlike hard masks, the token is still available, allowing for more fluent and flexible generation.

This **logit attenuation** approach enables BearMuzzle to shape style, tone, and behavior dynamically without modifying the base model or prompt.

---

### 🧠 Core Concepts and Mechanics

This section explains key mechanisms used in BearMuzzle for readers unfamiliar with logit manipulation and lightweight companion models.

---

#### 1. Logit Penalty and Attenuation vs Masking

- A *logit* is the raw score before softmax.
- BearMuzzle subtracts a penalty from selected logits to reduce their sampling probability.
- Unlike hard masks (which set logits to `-inf`), attenuation keeps the token available — just less likely.
- Think of it like dimming a light rather than switching it off.

---

#### 2. Penalty Vector Composition

BearMuzzle may combine multiple penalty vectors (e.g., tone, safety, domain) using:

- **Summation**: Adds each vector element-wise.
- **Max Aggregation**: Uses the strongest penalty per token.
- **Weighted Fusion**: Scales and sums different behaviors.

Example:

Tone Penalty: `[0.1, 0.0, 0.3]`  
Safety Penalty: `[0.0, 0.5, 0.1]`

- Summation: `[0.1, 0.5, 0.4]`
- Max: `[0.1, 0.5, 0.3]`

---

#### 3. Delta Logit Dataset Construction

To train a sidekick model:

- Run LLM twice: once with a behavioral prefix, once without.
- Record the logits for the same prompt/output.
- Compute the *difference* (delta) between the two outputs per token.

Example:

- Logits without prefix: `[1.5, 0.3]`
- Logits with prefix: `[1.0, 0.8]`
- Delta: `[-0.5, +0.5]`

This teaches BearMuzzle how the prefix modifies output behavior.

---

#### 4. Forced Token Emission / Injection

BearMuzzle can inject tokens (e.g., "In summary,") before resuming standard generation by:

- Setting the desired token’s logit high (or others to `-inf`) at that step.
- Softmax ensures the forced token is selected.

This maintains continuity in the sampling loop, without re-encoding the full context.

---

#### 5. Streaming and Time-to-First-Token

Why BearMuzzle avoids latency:

- Prefixing a prompt adds re-tokenization and warmup costs.
- BearMuzzle steers behavior *post-context*, avoiding changes to the original prompt.
- It computes penalties incrementally as tokens are generated.

---

#### 6. Ternary Weights vs Ternary Activations

- **Ternary Weights**: Model parameters use values from {-1, 0, +1}, allowing extreme compression (~1.58 bits).
- **Ternary Activations**: Intermediate neuron outputs also restricted to ternary values.
- BearMuzzle mostly assumes ternary weights, which are easier to optimize and compress.

---

#### 7. Context Matching and Tokenizer Fidelity

BearMuzzle and the main LLM must use the *exact same tokenizer*.

- If token IDs mismatch, the penalty will apply to the wrong token.
- Example: If "hello" is one token in the main model, but two ("he", "llo") in BearMuzzle, the steering breaks.

---

#### 8. Sparse vs Dense Penalty Vectors

- **Sparse**: Only a few token IDs receive non-zero penalties.
- **Dense**: All tokens are adjusted.

Sparse vectors are faster and consume less memory — ideal for real-time applications.

---

#### 9. Gradient Accumulation (for Low-Memory Training)

When batch size is too small for good gradients:

- Accumulate gradients over multiple mini-batches.
- Perform one optimizer step every N batches.

This simulates large batches on devices with limited RAM (e.g., 16GB Mac Mini).

---

#### 10. Logit Bias vs Logit Attenuation

OpenAI’s `logit_bias` applies static, fixed adjustments.

- BearMuzzle is dynamic — it computes penalties based on full context and output prefix at every generation step.
- More adaptive, fluent, and task-specific.

---


🗺️ High-Level Architecture

```
+----------------------+              +------------------------+
|                      |  Token N     |                        |
|     Main LLM         | -----------> |     Raw Logits         |
|    (GPU / llama.cpp) |              |   [51200 floats]       |
|                      |              +------------------------+
|                      |                         |
|                      |                         v
|                      |             +------------------------+
|                      |             |                        |
|                      |             |   BearMuzzle Sidekick  |
|                      |             |      (CPU, 1.5-bit)    |
|                      |             |                        |
|                      |             |   • Observes context    |
|                      |             |   • Computes bias mask  |
|                      |             |   • May trigger forced  |
|                      |             |     phrase injection    |
|                      |             +------------------------+
|                      |                         |
|                      |                         v
|                      |             +------------------------+
|                      |             |                        |
|                      |             |   Logit Adjuster       |
|                      |             |   (masking, biasing,   |
|                      |             |    forced tokens)      |
|                      |             |                        |
|                      |             +------------------------+
|                      |                         |
|                      |      Adjusted Logits    v
|                      |              +------------------------+
|                      |              |                        |
|                      | <----------- | Token N+1 Selection    |
|                      |              |                        |
+----------------------+              +------------------------+
```

This visual captures the flow between the two models and how logit guidance is applied.



🧠 BearMuzzle: A Companion Model for Real-Time LLM Guidance

BearMuzzle's CPU-optimized design ensures low-latency feedback by running inference alongside GPU token generation, leveraging asynchronous computation rather than serial prompt rewriting.

🧮 Why Attenuation Matters

Traditional logit masks often operate in binary: a token is either allowed (0) or completely suppressed (-inf). This rigid control can break fluency, produce unnatural phrasing, or entirely eliminate valuable tokens due to overaggressive filtering.

BearMuzzle, in contrast, uses a real-valued attenuation system:

- Penalties are floating-point scalars (e.g. 0.1–6.0) applied subtractively to logits.
- A mild penalty subtly discourages a token without forbidding it.
- Multiple influences (e.g. tone, safety, domain) can layer penalties in a nuanced fashion.
- This preserves fluency and diversity while applying directional guidance.

This technique avoids degenerate behavior common in hard-masking systems and improves compatibility with autoregressive sampling. It enables BearMuzzle to act as a bias shaper rather than a censor.

📌 Summary

This proposal outlines a system in which a secondary, ultra-lightweight language model runs concurrently with a larger LLM to adjust its logit mask in real-time. This enables dynamic, context-aware control over token selection during inference—using minimal CPU resources—without modifying the main model’s architecture or requiring retraining.

🎯 Objectives
	•	Real-time control over the token generation process of a large LLM
	•	Modulate token probabilities via logit attenuation (not hard masking)
	•	Offload this logic to a smaller CPU-based model for efficiency
	•	Enable steering, censorship, domain shaping, or stylistic adjustment without retraining

⚙️ System Architecture

Components
	•	Primary LLM (GPU-based): A standard transformer generating tokens, one step at a time.
	•	Secondary LLM (CPU-based): A fast, quantized model (~1.5-bit ternary model) that:
	•	Receives the same input context
	•	Optionally receives current output prefix
	•	Outputs a logit attenuation vector per token step

Workflow
	1.	The primary LLM begins a token-by-token inference loop.
	2.	At each step:
	•	The current context and prefix are sent to the CPU-based secondary LLM.
	•	The secondary LLM emits a logit attenuation vector (a sparse or dense penalty map).
	•	The logit penalties are subtracted from the primary LLM’s raw logits.
	•	Sampling or decoding proceeds as usual.

🎚️ Logit Attenuation vs Masking

Rather than using a binary hard mask, this system uses attenuation — real-valued penalties that adjust token probability without forbidding output.

Key Properties
	•	Penalty values are non-negative and subtracted from logits.
	•	Zero means “no change”
	•	Allows for nuanced guidance and avoids degeneracy

Example Penalty Table

Token	Penalty	Meaning
<profanity>	4.5	Strong discouragement
like	0.2	Mild stylistic softening
enterprise	0.0	No penalty
lol	1.0	Moderate informality penalty

Pseudocode

logits = big_model.forward(context)
penalties = small_model.forward(context, output_so_far)
logits -= penalties  # Apply attenuation before softmax
next_token = sample(logits)

This approach blends the flexibility of logit bias with the token-aware granularity of real-time soft control. Unlike static logit_bias settings in APIs or binary masks that entirely exclude tokens, attenuation enables expressive preference shaping — for example, discouraging profanity or informal slang without eliminating the possibility of nuanced use. This makes BearMuzzle suitable for domains where tone and subtlety matter.

🧩 Token Injection for Soft Steering


BearMuzzle can optionally inject a predetermined sequence of tokens—such as a reminder, directive, or soft prompt—at the start of inference. This sequence is injected directly into the output stream — bypassing sampling — to ensure exact reproduction of the prescribed phrasing. During this phase, token generation is forced rather than sampled, enabling precise behavioral seeding before resuming normal inference.

Key advantages:
- The injection occurs at the token level, avoiding prompt engineering or modified context.
- It does not delay the time-to-first-token (unlike prepending to the prompt).
- The position of the injected sequence can be tracked, and the tokens can be removed from the final output if desired, maintaining clean user-facing responses.
- Because it does not alter the model's prompt or context window, it avoids latency penalties associated with prepending behavior instructions — making it well-suited for streaming inference and low-latency applications.

📏 Boundary Tracking and Output Cleanup

Each injected sequence is accompanied by metadata recording its exact position in the generated output. This allows downstream components to:
- Identify injected spans unambiguously
- Optionally remove them from the final user-facing string
- Preserve alignment for downstream processing or auditing
- Enable token-aligned removal of injected sequences during post-processing, ensuring clean and faithful user-facing output while retaining full traceability for audit logs or interaction replay.

These positions are typically tracked using token start and end indices within the output stream. Since the tokens are deterministically emitted, this tracking is robust even in streaming contexts or multistep pipelines.

🔁 Midstream Injection

While the primary use case for injection is at the beginning of inference, BearMuzzle can also inject token sequences mid-generation. This allows for dynamic rerouting of behavior during long responses — for example, to restate safety instructions or pivot to a different tone.

Midstream injection follows the same forced-token mechanism, temporarily overriding the sampler to emit an exact sequence. After injection, BearMuzzle resumes attenuation-based steering seamlessly. These mid-output injections are useful in agentic workflows, interactive storytelling, or multi-turn dialogue control, where behavior may need reinforcement without modifying the initial prompt.

🔒 Implementation Detail:
The injected tokens are hard-coded into the output sequence by setting their corresponding logits to negative infinity for all non-target tokens during each forced step. This ensures deterministic emission of the intended phrase while preserving the integrity of the model's internal token state.


🔄 Post-Injection Attenuation

Once the forced injection sequence is completed, BearMuzzle transitions seamlessly into its standard operation mode: generating per-token logit penalties for the remainder of the inference loop. This allows the injected phrase to serve as a deterministic behavioral cue, after which the system continues to apply real-time, context-sensitive steering using logit attenuation. This transition is automatic and does not require external intervention or structural changes to the prompt or inference loop.

This allows BearMuzzle to softly shape the main model's behavior with high specificity and minimal infrastructure changes.

🕒 Latency Consideration

Unlike prompt engineering approaches that prepend instructions into the model’s context—thus increasing input length and computation—BearMuzzle’s injection strategy introduces no additional context-related latency. Because the injected tokens are emitted at the output stage (rather than parsed as part of the input), they preserve fast time-to-first-token performance. This makes BearMuzzle compatible with real-time chat interfaces, streaming decoders, and applications sensitive to generation delay.

🔍 Comparison with Prompt Engineering and Related Techniques

BearMuzzle's injection mechanism offers a distinct alternative to conventional prompt engineering:

- Unlike system prompts or prepended instructions, it does not consume context tokens.
- Unlike learned soft prompts or embeddings, it does not require retraining or context modification.
- Unlike OpenAI's `logit_bias`, it operates at each token step in real time and supports dynamic, per-token behavior.

This allows BearMuzzle to combine the flexibility of prompt-based steering with the precision of token-level control — without incurring the cost or complexity of modifying prompts, learning embeddings, or rebuilding token sequences. It is especially useful in latency-sensitive applications, multi-agent settings, or interactive use cases where dynamic control must be applied post-prompt.

⚠️ Challenges

Latency
	•	The small model must run in <5ms to avoid bottlenecks.
	•	Pre-generation or speculative lookahead may be needed.

Context Synchronization
	•	The BearMuzzle must consume the full token context, identical to the input fed to the main LLM.
	•	This guarantees behavioral fidelity, especially for prefix-simulation use cases.
	•	Tokenization must be consistent between both models.

Vocabulary Size
	•	Full vocab (e.g. 50k tokens) may be too large for direct output.
	•	Prefer sparse updates or top-k token attenuation.

Merging Multiple Influences
	•	Penalties from different sources (style, safety, persona) must be composable.
	•	Strategy: sum or max of penalties per token.

🏠 Model Details

What is a 1.5-bit LLM?

A 1.5-bit LLM uses ternary weight representation: -1, 0, and 1, which equates to 1.58 bits per parameter.

Benefits:
	•	Extremely compact memory footprint
	•	Efficient matrix ops using integer math
	•	Runs on general-purpose CPUs

Architectural Options:
	•	Tiny transformer with ternary quantization
	•	Shallow RNN or GRU with binary weights
	•	Token-attention MLP trained for penalty regression

⚡ Why 1.5-bit Models Are CPU-Friendly

The use of ternary weights (values -1, 0, or +1) is not only memory-efficient — it also leads to significantly faster inference on general-purpose CPUs. Here's why:

- **Multiplication-Free Math**: Instead of floating-point operations, matrix multiplications become additions, subtractions, or skips (for zeroes). This reduces instruction overhead and energy consumption.
- **Cache and Bandwidth Efficiency**: With fewer bits per parameter (1.58 bits vs. 8+), more weights fit into CPU caches, reducing memory bottlenecks — a major factor for CPU-bound inference.
- **Simplified Kernels**: Inference kernels can use integer math and even bitwise operations to pack and process ternary matrices efficiently. This aligns well with SIMD instruction sets (AVX, NEON).
- **On-the-Fly Decoding**: Packed ternary representations compress well, enabling fast loading and decoding at runtime — helpful for on-device or edge deployment.

These benefits make ternary LLMs a practical match for sidecar models like BearMuzzle, where real-time performance must be achieved using CPU resources alone.

📦 Memory Advantages of 1.5-bit Models

Ternary quantization (1.58 bits per parameter) yields substantial memory-related benefits, making it ideal for low-resource and edge inference:

- **Drastically Smaller Model Size**: A 100M parameter model fits in ~20MB, compared to ~400MB for float32. This makes on-device or embedded deployment practical.
- **Better Cache Utilization**: More parameters fit into CPU L1/L2 cache lines, reducing memory latency and boosting throughput — especially important on CPUs without specialized matrix hardware.
- **Efficient Bandwidth Usage**: Lower memory footprint translates to fewer memory fetches, improving inference speed on memory-bound systems.
- **Multi-Model Hosting**: With compact representations, several behavior-specific models can coexist in RAM, enabling fast runtime switching or ensemble shaping.
- **Streamable and Compressed**: Models can be stored in compressed formats with near-zero overhead, aiding container-based or mobile deployment scenarios.
- **Embedded-Agnostic**: While embedded environments benefit greatly from the reduced footprint, BearMuzzle is explicitly designed for general-purpose CPUs. This avoids hardware-specific constraints while still achieving near-embedded efficiency.

These memory savings compound the compute efficiency gains to make 1.5-bit ternary models an ideal sidekick for real-time logit modulation on CPU-bound systems.

🔍 Precision Notes on Intermediary Layers

While BearMuzzle uses 1.58-bit ternary quantization for model weights, the bit depth of intermediary computations — such as activations, attention scores, and normalization outputs — can vary and may influence both runtime speed and precision.

✅ Current Practice:
- **Weights**: Stored as ternary values (-1, 0, +1), using ~1.58 bits per parameter.
- **Activations**: Typically computed in int8, int4, or float16.
- **Attention and Norm Ops**: May require float16 or float32 temporarily due to softmax or residual scaling operations.

This hybrid-precision approach balances efficiency and model fidelity:
- Low bit-width activations reduce RAM usage and improve matrix op speed.
- Mixed precision preserves expressive power where needed.

🧪 Potential Optimizations:
- Explore ternary activations or integer-only attention for even leaner inference loops.
- Support selectable quantization tiers (e.g., "low-latency", "balanced", "high-precision") to match application constraints.

🧠 Why It Matters:
Even when weights are compact, intermediary values dominate the working memory during inference. Lowering activation precision helps maintain speed and memory efficiency, particularly for real-time inference on CPU.
Memory Considerations

Memory usage is not strictly limited to <100MB, but minimal memory footprint is a target for wide deployment. A tiered design allows for:

RAM Budget	Possible Architecture	Tradeoffs
<32MB	Tiny MLP, 1–2 layers	Extremely fast, very shallow
~64MB	2–4 layer ternary transformer	Good balance
128–256MB	6+ layers, wider heads	Higher latency, better semantics
>256MB	Near full LLM-lite	CPU may bottleneck

💰 Training Cost Considerations

Given that the BearMuzzle must process the full input context, training data must simulate full-context behavior accurately. This affects training cost as follows:

Training Dataset Construction
	•	For each training sample:
	•	Run the main LLM twice on a prompt:
	•	Once with a behavioral prefix (e.g., “Answer formally.”)
	•	Once without
	•	Extract per-token logits and compute the delta between the two runs
	
	This delta captures the directional influence of the behavioral prefix on the main LLM's output distribution. BearMuzzle learns to reproduce this delta in real time as a logit attenuation vector, allowing it to steer the main model as though the prefix were present — without modifying the prompt or context. Over many examples, the model generalizes from specific prefix behaviors to broader stylistic or safety tendencies.
	•	With full context (~4K tokens), this means that for each output token, BearMuzzle must learn from a token-wise delta over the entire input context
	•	If 10K prompts are used × 10 behaviors × 50 token generations = ~5 million labeled deltas
	•	These runs can be parallelized, but compute-intensive


## 💰 Training Cost Speculation (Example for Qwen 3-0.5B)

Training a BearMuzzle-compatible sidekick model using Qwen 3-0.5B dramatically reduces compute and memory requirements while preserving compatibility with larger Qwen 3 models.

- ✅ The model is just 0.5B parameters, making it suitable for training on modern laptops or Mac Mini-class devices.
- 🧠 The full vocabulary and token output space is consistent across Qwen 3 family models, so training is transferable to Qwen 3-7B and 14B without remapping.
- 🧪 Example batch sizes of 2–4 can be used with context windows of 2k–4k on machines with **16GB RAM**.
- 🔁 Epoch throughput is rapid: thousands of examples can be cycled through in a few hours.
- ⏱️ A sidekick model trained on 25k examples could be fine-tuned in **under 8 hours** on a Mac Mini (M4 or similar) or a 1× T4/A10G instance.

This makes small-Qwen sidekicks extremely viable for rapid prototyping and usable production steering tools — not just experimental scaffolds.

📝 Summary:
	•	Training the BearMuzzle is relatively light
	•	Generating its training data using full-context inference is the more expensive phase
	•	Training is feasible on consumer-grade GPUs or Mac-class CPUs, though training time may span multiple days
	•	Training is feasible on consumer-grade GPUs, even when using 4K token contexts

➡️ In future iterations, we may explore compressing or summarizing the context to reduce cost, but fidelity-first training requires matching the main model’s full prompt.


✅ Conclusion on Context Matching
	•	The BearMuzzle’s fidelity depends on matching the token-level behavior of the target LLM
	•	Therefore, it must consume the full context window, with consistent tokenization
	•	This enables accurate simulation of behaviors encoded in long-range prompts or prefixes
	•	We note the associated increase in training and inference costs, and accept them as tradeoffs for behavior-preserving alignment

---

### 🪶 Small Models, Large Influence: Efficient Training for Practical Sidekicks

BearMuzzle leverages extremely compact models — often 1.5-bit ternary architectures with low parameter counts — to act as real-time logit modulators. These models are not mere prototypes; they are highly practical for production use due to their size, speed, and generalization efficiency.

**Key advantages:**

- **Maximized Training Throughput**: Tiny models allow for massive-scale training on diverse behavioral deltas within a fixed compute budget. More examples can be processed per GPU hour, yielding broader generalization from richer signal.
- **Low Latency at Inference**: These models are designed for sub-5ms inference on commodity CPUs, allowing real-time use in agent loops or chat interfaces.
- **Behavioral Modularity**: Multiple compact models (e.g., tone, domain, safety) can coexist in RAM and contribute independently to the final penalty vector — enabling dynamic fusion.
- **Transferability Across Models**: When token vocabularies align, the same compact sidekick model can steer multiple LLM variants — maximizing ROI on training effort.
- **Edge Deployment**: Small models are viable for edge scenarios without dedicated GPU access, expanding the reach of adaptive inference systems.

Training these models on rich, high-coverage datasets — generated by behavioral deltas from large LLMs — yields a data-to-weight ratio that promotes robust generalization. The compactness does not limit usefulness; in fact, it enhances practicality.

This strategy contrasts with traditional large-model pretraining and opens a new path for behavior shaping: **train small models on rich token-wise behavioral guidance** and apply them widely across LLMs that share a vocabulary.

🚀 Next Goals Toward Publication

To elevate this project to a level suitable for arXiv or conference submission, the following milestones must be reached:

1. Working Prototype
	•	Modify llama.cpp or compatible runtime to accept external logit penalty vectors
	•	Implement IPC or hook system to inject penalties per token step
	•	Ensure synchronization between BearMuzzle and main LLM context
	•	Patch the llama.cpp inference loop (e.g., around `llama_sample()` or `llama_logits`) to subtract externally computed logit penalties before sampling. This provides a minimal viable integration point with almost no disruption to the LLM’s internal architecture.

2. BearMuzzle Model
	•	Train a compact model (e.g., ternary transformer or sparse MLP)
	•	Demonstrate <5ms inference per token on CPU
	•	Support at least one behavioral profile (e.g., formal tone)

3. Dataset and Training
	•	Generate a delta logits dataset using main LLM with/without prefixes
	•	Provide token-aligned examples and penalty vectors
	•	Document model training settings and accuracy on held-out samples

4. Evaluation and Comparison
	•	Produce LLM outputs under:
	•	No control
	•	Prefix-based prompt engineering
	•	Static logit bias
	•	BearMuzzle steering
	•	Metrics: BERTScore, BLEU, human preference, token-level match
	•	Include latency and memory benchmarks

5. Formal Method Description
	•	Define BearMuzzle mathematically as a delta predictor
	•	Express training loss and model structure
	•	Discuss composability of multiple behavior targets

6. Baseline Analysis
	•	Compare to prior work:
	•	PPLM (Plug and Play LM)
	•	Classifier-guided decoding
	•	OpenAI logit_bias
	•	Show how this approach is faster, lower-latency, and non-invasive

7. Write and Submit Paper
	•	Structure: Intro, Related Work, Method, Training, Experiments, Results
	•	Submit to arXiv or venue such as NeurIPS, ICML, or EMNLP

These goals constitute the critical path from experimental design to public dissemination.

🔧 llama.cpp makes it particularly straightforward to intercept token-level logits due to its transparent sampling loop and exposed API surface. This allows for rapid prototyping of BearMuzzle behaviors with minimal risk to core model stability.

🧠 Use Cases Beyond Censorship

BearMuzzle is not limited to filtering or content suppression. Its ability to softly influence token selection opens up a diverse range of applications:

- 🗣️ Tone Steering: Maintain formality, enthusiasm, or politeness without brittle prompt phrasing.
- 🧾 Domain Alignment: Encourage legal, technical, or medical phrasing based on task context.
- 👤 Persona Shaping: Enforce consistent speaking style for virtual characters or brand voices.
- ⚠️ Safety Enforcement: Reduce likelihood of inappropriate or off-topic content generation.
- 🔁 Prompt Reweighting: Emphasize parts of the prompt (e.g., style guide, examples) dynamically during generation.
- 🪞 Behavior Simulation: Inject temporary traits like sarcasm, humor, or formality for adaptive output control.
- 🧭 Dynamic Rerouting: Intervene mid-output to reinforce or shift conversational direction without restarting the model or altering initial prompts — ideal for interactive agents or multi-turn reasoning tasks.

- 🧠 Chain-of-Thought Refinement: Insert hints or scaffolding mid-sequence to encourage multi-step reasoning or problem-solving, especially useful in educational tools or planning agents where guidance should adapt over time.

🧩 Multi-Behavior Composition Strategies

BearMuzzle enables composable behavioral shaping by allowing multiple influence sources to contribute to the final logit penalty vector at each token step.

Strategies for combining multiple behaviors include:
- **Summation**: Additive combination of penalty vectors (e.g., tone + domain + safety).
- **Max Aggregation**: Use the highest penalty per token from all sources.
- **Weighted Fusion**: Learn a blending function to combine multiple influences dynamically.

Example:
- Tone guidance suggests discouraging slang.
- Safety model adds penalties for offensive terms.
- Domain alignment favors formal vocabulary.

These can be composed to produce a blended penalty vector that captures multiple overlapping constraints. This supports more nuanced and situation-aware generation without requiring explicit prompt changes or task switching.

BearMuzzle's compositional capability is particularly powerful in dynamic environments where task boundaries blur — such as multi-agent systems, open-ended assistants, or adaptive storytelling — allowing real-time interpolation between behavioral modes without hard transitions or re-prompts.
This flexibility enables dynamic context interpretation, such as progressively increasing the influence of a safety layer as the conversation veers into sensitive topics, or dialed modulation of tone as user mood shifts — all without restarting the inference loop.
Another direction involves reactive shaping — where BearMuzzle monitors generated output in real time and adjusts its attenuation strategy based on token-level trends or user feedback. For instance, if the generated text starts drifting off-topic, BearMuzzle could increase penalties for tangential branches or re-weight the original task signal. This creates a feedback loop that helps maintain alignment in evolving or ambiguous prompts.

---

### 🔄 Learned Steering from Linter Feedback

BearMuzzle’s linter integration does more than respond after mistakes — it can anticipate and actively shape generation patterns.

The sidekick model observes linter warnings over the course of inference. If similar mistakes recur (e.g., `== None`, mutable default arguments), it can begin to apply logit penalties to problematic tokens and slight boosts to preferred alternatives. This transforms BearMuzzle from a post-hoc annotator into a real-time behavioral coach.

#### 🧠 Example

**Without Steering:**
```python
def foo(bar={}):
    # ⚠️ Avoid mutable default arguments
```

**With Steering (later in same session):**
```python
def foo(bar=None):  # (no linter warning needed)
```

The sidekick has learned to steer away from patterns that previously triggered corrective comments. This anticipatory biasing reinforces good coding habits without altering the core model.

Such behavior can be trained offline using context+code pairs with and without feedback prompts, forming a lightweight habit-shaping loop that improves output across longer sessions.

---

#### 12. 🧬 Simulated Persona Merging

BearMuzzle enables multiple behavior-altering penalty vectors to be blended in real time, enabling composite personas.

Example scenario:

- **Tone Penalty Vector**: discourages aggressive or curt phrasing.
- **Safety Penalty Vector**: suppresses risky or inappropriate tokens.
- **Domain Penalty Vector**: promotes technical or field-specific vocabulary.

These vectors are combined using a weighted strategy:

```python
final_penalty = 0.5 * tone + 0.3 * safety + 0.2 * domain
```

This allows rich behavior synthesis. For example:

> “Respond like a helpful attorney with medical caution and polite tone.”

By adjusting weights dynamically, BearMuzzle can support fluid shifts in style and alignment without interrupting inference.

---

### 🔁 Model Transfer and Tokenizer Compatibility

BearMuzzle's logit attenuation vectors are tied to the output vocabulary of the main language model. These vectors operate over a specific token index mapping, meaning the compatibility of BearMuzzle with other models depends on whether they use the same tokenization scheme and output token IDs.

In the context of open-source models compatible with local inference (e.g., those run via `llama.cpp`), transferability of a trained BearMuzzle model is straightforward when:

- The large model uses the **same tokenizer** and **vocabulary index order**.
- The token IDs for commonly used tokens remain identical across models.

✅ **Directly Compatible Examples**:
- LLaMA 2 7B and LLaMA 2 13B
- LLaMA 3 variants that preserve the tokenizer spec
- Mistral and Mixtral models when tokenizer reuse is confirmed

❌ **Incompatible Without Retraining**:
- LLaMA to Falcon (different tokenizer specs)
- Mistral to T5 or BLOOM variants
- Any cross-family transfer where tokenizer internals differ

If token IDs differ, then BearMuzzle's output will misalign — applying penalties to unintended tokens. To reuse in such cases, a retraining step is required using the same behavioral delta procedure, but with the new model’s tokenizer and context pipeline.

📌 Recommendation:
- Prefer training BearMuzzle on the most common or stable tokenizer spec within your target deployment ecosystem.
- Maintain tokenizer documentation and versioning alongside model checkpoints to ensure reproducibility.

This approach supports seamless reuse of BearMuzzle across model variants that share a common token matrix, enabling efficient steering without retraining in many practical deployments.
### 🧑‍💻 Training on Consumer Hardware (e.g., Mac Mini M4)

While high-end GPUs accelerate data generation and training, compact 4-bit sidekick models can be trained on modern consumer machines with careful constraints.

Example: **Mac Mini M4 (2024), 16GB RAM**

| Factor                    | Value                                           |
|--------------------------|-------------------------------------------------|
| Model Size               | ~100M params, 4-bit (~50MB)                     |
| Training Hardware        | Mac Mini M4, 16GB unified memory, CPU+Metal GPU |
| Estimated Throughput     | ~2–5 samples/sec (tiny models only)             |
| Epoch Time               | 35–87 hours (1 epoch), 70–350 for full training |
| Training Duration        | 3–14 days continuous (for 2–4 epochs)           |
| Feasibility              | ✅ Reasonable for a single-user, long-running job |
| Practicality             | ✅ Yes, with patience and memory-aware batching  |

Tips:
- Keep batch sizes small (e.g., 4–8 samples)
- Use gradient accumulation to simulate larger batches
- Preprocess delta vectors in advance to reduce token load
- Checkpoint frequently to allow for restarts

This enables solo developers to build viable BearMuzzle models on consumer hardware.
---

### 🧪 Related Work and Research Alignment

Recent research in the field of LLM steering has explored concepts similar to BearMuzzle’s logit modulation approach. These works validate the design direction and highlight BearMuzzle’s unique external and token-level strategy.

#### 🟢 [Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs](https://arxiv.org/abs/2505.20309)
This paper introduces compact controller networks that modify internal activations using layer-wise scaling to steer model behavior.

- **Similarity**: Uses lightweight, real-time control without altering the base model weights.
- **Difference**: Operates on hidden layer activations instead of final logits.
- **Relevance**: Demonstrates the feasibility of external alignment using auxiliary modules.

#### 🟢 [Steering Large Language Models with Feature-Guided Activation Additions](https://arxiv.org/abs/2305.10967)
FGAA uses sparse, interpretable activation modifications derived from autoencoder-encoded features to steer output.

- **Similarity**: Uses structured, interpretable vectors to guide output behavior.
- **Difference**: Targets activations, not logits; focuses on inner-state perturbation.
- **Relevance**: Supports the notion that behavioral deltas can be encoded as lightweight overlays.

#### 🟢 [Identifiable Steering via Sparse Autoencoding of Multi-Concept Shifts](https://arxiv.org/abs/2306.17045)
This work identifies steering vectors using sparse autoencoders to capture and inject multiple aligned behavioral concepts.

- **Similarity**: Supports composable alignment vectors to modulate style or tone.
- **Difference**: Operates in the activation space; requires separate encoder/decoder training.
- **Relevance**: Suggests sparse vector fields can encode domain-specific behavior cues.

#### 🟢 [Steering Large Language Models Using Conceptors](https://arxiv.org/abs/2402.09691)
Applies conceptor matrices to project activations into aligned subspaces—representing constraints like helpfulness or harmlessness.

- **Similarity**: Supports real-time, modular steering without model retraining.
- **Difference**: Uses linear algebraic projection; focused on high-level behaviors.
- **Relevance**: Shows behavioral priors can be enforced at runtime through lightweight overlays.

#### 🟢 [ExpertSteer: Expert Model Alignment Control via Intermediate Fusion](https://arxiv.org/abs/2404.03929)
Proposes combining multiple smaller "expert" models via fusion layers to shape LLM behavior mid-inference.

- **Similarity**: Uses external models to steer generation dynamically.
- **Difference**: Requires fusion interfaces and internal access; not logit-targeted.
- **Relevance**: Supports the trend of distributed, modular influence during inference.

---


BearMuzzle remains unique in that it:
- Acts externally at the logit level.
- Works with quantized or CPU-executable "sidekick" models.
- Requires no access to hidden states or model retraining.

---

### 🧭 Comparison Table: BearMuzzle vs. Activation-Based Steering

| Feature                     | BearMuzzle                                  | Related Works                           |
|----------------------------|---------------------------------------------|-----------------------------------------|
| **Manipulation Level**     | Logit-level attenuation/injection           | Latent activations / hidden layers      |
| **Runtime Overhead**       | Lightweight CPU-sidekick (quantized)        | Lightweight controller nets             |
| **Training Inputs**        | Delta logits between prefix/no-prefix       | Concept-encoded activations             |
| **Composability**          | Designed for multi-vector blending          | Usually discrete concept embeddings     |
| **Hardware Requirements**  | Minimal, CPU-executable                     | Often GPU-optimized                     |
| **Integration Layer**      | External to model inference loop            | Often requires access to model internals|

---

### 📌 Additional Implications

- **No llama.cpp Modifications Required**:  
  BearMuzzle interacts with existing APIs such as logit biasing or forced token sequences, requiring no patching of llama.cpp internals.

- **Decoupled Development**:  
  Because the penalty vectors are trained externally, BearMuzzle models can evolve independently from the LLM they steer—provided the tokenization schema is shared.

- **Cross-Model Applicability**:
BearMuzzle sidekick models can be trained on drastically smaller or highly quantized versions of a given LLM and later applied to larger or more accurate models of the same architecture. As long as the token vocabulary and inference APIs remain consistent, this approach allows for low-cost, scalable training with generalization across model tiers and future generations.

- **Immediate Integration**:  
  Penalty overlays can be swapped in or out mid-inference without reinitializing or recompiling the base model.

- **Stackable Steering**:  
  This logit-layer mechanism can operate alongside activation-based controllers, supporting hybrid modulation pipelines.

---

---

## 🧩 Sidekick Compatibility Matrix for Open Models

This matrix summarizes compatibility between major open-weight LLM families and BearMuzzle-style sidekick inference control.

| Model Family | Tokenizer Shared Across Sizes | Compatible with Sidekick Training at Small Scale | llama.cpp Inference Supported | Notes |
|--------------|-------------------------------|--------------------------------------------------|-------------------------------|-------|
| **Qwen 3**   | ✅ Yes                         | ✅ Yes                                           | 🟡 Partial (some patching needed) | Ideal candidate; consistent tokenizer and strong small models |
| **LLaMA 2**  | ✅ Yes                         | ✅ Yes                                           | ✅ Yes                        | Most thoroughly supported in `llama.cpp`; baseline model for BearMuzzle |
| **LLaMA 3**  | ✅ Yes                         | ✅ Yes                                           | ✅ Yes                        | Large context length models may require tuning sidekick context assumptions |
| **Mistral**  | ✅ Yes                         | ✅ Yes                                           | ✅ Yes                        | Good choice; fast inference and stable quantized variants |
| **Gemma**    | ✅ Yes                         | ✅ Yes                                           | 🟡 Partial                    | Limited community support; still promising |
| **Falcon**   | 🔶 Unclear tokenizer consistency | 🔸 Uncertain                                     | 🔸 Less common in llama.cpp   | May require additional effort to align output token IDs |
| **Command-R**| ❌ No (not open)               | ❌ No                                            | ❌ No                         | Not usable under BearMuzzle model without closed API access |

---

This compatibility table is updated as new open-weight families emerge or get `llama.cpp` support. All rows assume the use of shared tokenizer vocabularies and RoPE or ALiBi-compatible positional encodings for successful logit shaping.

---

## 🧭 Phase Two Development Prospects

The following ideas expand BearMuzzle’s capabilities by leveraging existing infrastructure—namely, the logit manipulation hook, external sidekick model, and runtime integration through llama.cpp’s token loop. None of these require internal modification to the main model.

### 🔄 Real-Time Modulation Enhancements

- **Drift Detection**  
  Detect deviation from intended tone, domain, or persona and dynamically suppress off-topic or irrelevant token groups.

- **Context-Aware Ending Bias**  
  Steer generation toward punctuation or semantic closure when nearing a logical wrap-up phase.

- **Soft Reminder Phrase Injection**  
  Insert a predetermined phrase mid-sequence without blocking or introducing delay, and mark it for post-removal. Useful for safety, disclaimers, or reframing prompts.

- **Stochasticity Damping in Sensitive Contexts**  
  Identify emotionally charged or safety-critical zones and reduce output entropy through logit scaling.

- **Topic Boundary Detection**  
  Recognize when a topic shift occurs and softly bias toward transitional phrases (e.g., “Next, let’s...” or “To summarize…”).

### 🧠 Structure and Style Steering

- **Token Cage Completion**  
  Gradually increase penalty outside a targeted phrase to encourage convergence without forcing it explicitly.

- **Rolling Redundancy Avoidance**  
  Penalize tokens that would repeat recent bigrams or trigrams to prevent looping or verbose filler.

- **Mid-Stream Persona Shifting**  
  Change behavioral tone or domain vector mid-output based on context or external signal (e.g., from a UI toggle or control code).

- **Token-Time Awareness**  
  Adjust generation pacing or tone based on the real-time duration of inference, useful for responsive UIs.

- **Interrupt-Aware Smoothing**  
  When running in streamed environments, steer generation toward natural stopping points in anticipation of possible interruption.


### 🔧 Meta-Control Extensions

- **User-Controlled Persona Blending**  
  Dynamically compose behavior vectors (e.g., 40% safety + 30% helpfulness + 30% brevity) via UI or API input.

- **Adaptive Penalty Switching**  
  Replace or blend sidekick models or penalty overlays at runtime in response to detected dialogue phases or external events.

---

## 🧰 Tool Use and Feedback Integration

BearMuzzle supports integration with tool-using LLM runtimes, enabling dynamic communication between the main LLM and the sidekick biasing model. This opens a powerful channel for both proactive and reactive inference shaping.

### 🔄 Two-Way Interaction Model

**1. LLM-Initiated Sidekick Control**

The main model can explicitly invoke tools to update the sidekick’s behavior mid-inference. For example:

```json
{
  "tool_call": {
    "name": "adjust_bias",
    "arguments": {
      "discourage": ["eval", "exec"],
      "encourage": ["safe_eval"]
    }
  }
}
```

This allows dynamic logit penalty/boost configurations during runtime, scoped as needed.

**2. Sidekick Reactivity to Tool Usage**

The sidekick can observe emitted tokens and recognize tool use patterns (e.g., a `search_web` call or a `ChatCompletion` API invocation). It can then apply context-specific shaping:

| Tool Used         | Sidekick Reaction                              |
|------------------|--------------------------------------------------|
| `search_web`     | Encourage synthesis-style phrasing               |
| `summarize_text` | Boost structured output (bullet points, headers) |
| `get_docstring`  | Inject reminders or format hints                 |
| `check_style`    | Trigger stricter linter overlays                 |

### 🛠️ Example Control Commands

The following represent a conceptual API between the LLM and the sidekick:

- `set_bias({token: weight})`
- `inject_reminder(text)`
- `learn_pattern({bad: good})`
- `apply_preset("security")`
- `scope_bias(N_tokens)`
- `pause_masking()` / `resume_masking()`

These can be implemented in any host framework that supports tool calls or plugin hooks.

### 🔮 Future Work

Planned capabilities include:
- Sidekick awareness of user-defined modes (e.g. `"developer_tone"`, `"factual_only"`)
- Bidirectional adapters for runtime frameworks like OpenAI Assistants, LangChain, and Semantic Kernel
- Contextual policy enforcement layers (e.g., safety, verbosity, branding)

BearMuzzle is positioned as a real-time shaping layer for inference behavior — modifiable by the main model itself.
---

## 🧹 Inference-Time Correction with Post-Hoc Cleanup

BearMuzzle embraces the immutability of LLM token emission — and leverages it by introducing a correction flow that marks and repairs errors during inference while preserving a clean final output through post-processing.

### 🔄 The Constraint

In transformer-based LLMs, once a token is emitted, it **cannot be retracted or replaced**. Even if the model realizes it made a mistake mid-sentence, it must continue building from that token onward.

### 🧠 BearMuzzle Strategy

BearMuzzle sidekick model can intervene to:

1. Allow the LLM to emit an incorrect or suboptimal line.
2. Immediately follow up with a **forced error marker** (e.g., `# ❌ Logic error: comparison is always true`).
3. Guide the LLM to emit a **corrected version** of the flawed line.
4. Optionally bracket this region for post-inference cleanup.
5. During final output, a simple transformation removes the flawed line and correction hint, preserving only the corrected content.

This enables **regenerative correction** without breaking the inference loop.

### 🧪 Example: Correcting a Code Snippet

**During inference:**

```python
if x == None:
# ❌ Prefer 'is None' for comparison
if x is None:
```

**After cleanup:**

```python
if x is None:
```

This behavior mirrors how a human might revise code during typing — thinking out loud, marking flaws, and rephrasing — but produces a clean output at the end.

### ⚙️ Cleanup Strategies

Post-inference, the system can:

- Detect lines starting with `# ❌` or custom markers.
- Identify their adjacent correction block.
- Remove both the flawed line and the hint.
- Retain the corrected form only.

This stage is entirely modular and can be configured per use case (e.g., preserve annotations for audit trails, strip for clean output, etc.).

### 🛠️ Applications

- LLM code generation with automatic rewrites
- Context-aware error recovery without hallucination penalties
- Editable audit trails in educational tools
- Use in CI pipelines for formatting and correction

### 🔍 Example Use Flow

| Step | Output |
|------|--------|
| LLM emits flawed line | `if user == None:` |
| Sidekick appends comment | `# ❌ Use 'is None' instead` |
| Sidekick forces fix | `if user is None:` |
| Post-pass strips old lines | Final: `if user is None:` |

### 🔄 Why It Works

This flow acknowledges that:
- You can't undo a token.
- But you can annotate, correct, and erase it after the fact.
- All while keeping the LLM in its natural autoregressive loop.


---

### 📚 Connections to Speculative Decoding in LLM Inference

BearMuzzle's post-hoc cleanup strategy echoes the mechanisms found in speculative decoding research. One such example is described in:

> **LLM Inference Unveiled: Survey and Roofline Model Insights**  
> [arXiv:2405.18080](https://arxiv.org/abs/2405.18080)

This work explains how small "draft" models can generate tokens that are conditionally accepted or rejected by the main LLM:

> "The small model can speculate and generate draft tokens ... Some of the tokens are rejected, so the large model asks the small model to ‘roll back’ these wrong tokens and resume speculating."

This process aligns conceptually with BearMuzzle’s correction approach:
- Allow potentially flawed tokens to be emitted.
- Immediately flag them with error marker tokens or comments.
- Force corrected versions to follow.
- Strip marked spans after inference to produce a clean final result.

#### ✅ Why This Matters

- Validates BearMuzzle’s token-stream-respecting correction method.
- Confirms feasibility of approximate-first, corrected-later inference.
- Shows alignment with emerging efficiency paradigms like speculative decoding.
- Provides an opportunity to benchmark correction accuracy vs. speculative draft acceptance rates.

This enhances BearMuzzle’s conceptual grounding and connects it to active work in efficient LLM inference.
---

## 🧠 Bracketed Reasoning and Deliberate Planning via Sidekick Guidance

BearMuzzle enables structured reasoning behavior in LLMs through indirect control of token flow — even in the absence of agent-native capabilities like function calling or intermediate steps.

By shaping the likelihood of reasoning-related tokens (e.g., `<think>`, `</think>`, `Answer:`), the sidekick model can enforce boundaries and rhythms in the model's thought process.

### 🧩 Key Patterns Enabled

#### 1. **Encouraging a Thinking Block Start**

If the model begins responding without deliberation, BearMuzzle can:
- Detect the lack of a `<think>` block.
- Strongly bias the emission of a `<think>` token.
- Suppress direct answer phrases (`Answer:`) until reasoning has occurred.

> 📌 Outcome: Improves planning and discourages impulsive responses.

#### 2. **Ending a Reasoning Phase**

BearMuzzle can nudge the model to:
- Emit `</think>` after sufficient tokens or heuristic cues (e.g., repetition or tautologies).
- Transition naturally into an answer or action phase.

> 📌 Outcome: Prevents runaway reasoning, keeps outputs efficient and bounded.

#### 3. **Reopening Thought After Tool Use**

If the LLM uses a tool (e.g., database lookup or calculator), BearMuzzle can:
- Detect tool result lines.
- Enforce a second `<think>` block before finalizing the answer.

> 📌 Outcome: Encourages multi-phase reasoning in complex workflows.

#### 4. **Bracketing via Forced Tokens with Cleanup**

When paired with BearMuzzle's post-inference cleanup layer:
- Forced `<think>...</think>` blocks can be inserted inline.
- These spans can be stripped before final output.
- Or retained for auditability.

> 📌 Outcome: Enables verbose internal reasoning while keeping clean UI results.

### 🔧 Runtime Implementation

- Token selection adjusted by sidekick during inference loop.
- LLM output remains uninterrupted, preserving native autoregressive flow.
- Thinking regions are tracked for optional trimming or scoring.

### 🛠️ Advanced Use Cases

- Chain-of-thought QA
- Agent-style planning in a flat stream
- Code analysis with multi-step justification
- Policy recommendation systems that think aloud

This structure allows BearMuzzle to simulate "thinking" behavior in local models, making the output more explainable, structured, and aligned with advanced prompting strategies.


---

## 🧰 Development Strategy: Python + Rust

BearMuzzle is structured for agile research with a stable deployment path:

### Phase 1: Python for Prototyping

We use `llama-cpp-python` to:

- Rapidly experiment with sidekick masking loops
- Validate logit shaping and prefix injection
- Iterate on integration with tool-use or linter feedback

This allows for quick testing and evaluation before deeper optimization.

### Phase 2: Rust for Implementation

We will implement the final system in Rust using:

- Direct FFI access to `llama.cpp`
- Custom inference loops with exposed logits
- Efficient multi-threaded coordination (main + sidekick models)
- CPU-first runtime performance and lightweight deployment

This approach ensures high portability, low latency, and robust resource control.

### 🧬 Python Bindings from Rust

The Rust implementation will expose bindings using tools like `pyo3` or `maturin`, enabling:

- Python access to Rust-accelerated inference loops
- Drop-in compatibility with `llama-cpp-python`
- A smooth bridge between research code and production logic

This layered architecture allows researchers, builders, and embedded system engineers to collaborate without friction.