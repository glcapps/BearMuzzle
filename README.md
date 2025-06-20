# BearMuzzle
A lightweight CPU-side model that dynamically adjusts the logits of a larger LLM during inference — for real-time steering, behavioral shaping, or stylistic control without modifying the base model.

🧠 BearMuzzle: A Companion Model for Real-Time LLM Guidance

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

Cost Impact

Phase	Resource	Cost (est.)
Data generation	700 GPU hours	~$350–$800
Dataset storage	~5GB	Negligible
BearMuzzle training	<10 GPU hours	~$10–$30

📝 Summary:
	•	Training the BearMuzzle is relatively light
	•	Generating its training data using full-context inference is the more expensive phase
	•	Training is feasible on consumer-grade GPUs, even when using 4K token contexts

➡️ In future iterations, we may explore compressing or summarizing the context to reduce cost, but fidelity-first training requires matching the main model’s full prompt.

✅ Conclusion on Context Matching
	•	The BearMuzzle’s fidelity depends on matching the token-level behavior of the target LLM
	•	Therefore, it must consume the full context window, with consistent tokenization
	•	This enables accurate simulation of behaviors encoded in long-range prompts or prefixes
	•	We note the associated increase in training and inference costs, and accept them as tradeoffs for behavior-preserving alignment

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