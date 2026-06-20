"""
Lab 2: Enhancing LLMs using RAG and Hugging Face
=================================================
Topic: Retrieval-Augmented Generation (RAG) with Dense Passage Retrieval (DPR)
       and GPT-2 as the generative model.

This lab demonstrates:
  - Dense Passage Retrieval (DPR) for context retrieval using Hugging Face
  - GPT-2 for text generation, both without and with retrieved DPR contexts
  - Observations on quality differences between direct vs. augmented generation
  - Exercise: Tuning generation parameters (max_length, min_length,
    length_penalty, num_beams) and analysing their effects

Key observations (reproduced from lab notes):
  * Without DPR: GPT-2 relies solely on pretrained knowledge → generic answers.
  * With DPR:    Retrieved contexts greatly improve accuracy and detail.
  ⟹ Combining retrieval + generation is more effective than generation alone.

Author notes:
  Based on IBM Skills Network lab material.
  Extended with full implementations, commentary, and parameter-tuning exercise.

Prerequisites:
  pip install transformers torch faiss-cpu

Note on compute:
  DPR and GPT-2 are large models. A GPU is recommended but not required.
  Expect 2–5 minutes for model loading on CPU.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  (small passage corpus for demonstration)
# ─────────────────────────────────────────────────────────────────────────────
PASSAGES = [
    # Paris / Eiffel Tower
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. "
    "The tower stands 330 metres tall and was the world's tallest man-made structure for 41 years.",

    # Python programming
    "Python is a high-level, general-purpose programming language. "
    "Its design philosophy emphasises code readability with the use of significant indentation. "
    "Python is dynamically typed and garbage-collected, and supports multiple programming paradigms.",

    # Climate change
    "Climate change refers to long-term shifts in temperatures and weather patterns. "
    "These shifts may be natural, but since the 1800s, human activities have been the main driver, "
    "primarily due to the burning of fossil fuels, which produces heat-trapping gases.",

    # Photosynthesis
    "Photosynthesis is the process by which green plants and some other organisms use sunlight, "
    "water, and carbon dioxide to produce oxygen and energy in the form of glucose. "
    "It takes place mainly in the chloroplasts using the green pigment chlorophyll.",

    # Machine learning
    "Machine learning is a branch of artificial intelligence concerned with the development "
    "of algorithms that allow computers to learn from and make predictions or decisions based on data. "
    "Supervised, unsupervised, and reinforcement learning are its three main paradigms.",

    # RAG itself
    "Retrieval-Augmented Generation (RAG) is an AI framework that combines a retrieval model "
    "with a generative model. The retriever fetches relevant documents from a knowledge base, "
    "and the generator uses those documents as context to produce more accurate, grounded answers.",
]

QUERY_EXAMPLES = [
    "When was the Eiffel Tower built?",
    "What is machine learning?",
    "How does photosynthesis work?",
    "What causes climate change?",
    "What is Retrieval-Augmented Generation?",
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DPR MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_dpr_models(device: str = "cpu"):
    """
    Loads DPR context encoder and question encoder from Hugging Face.

    Returns:
        ctx_encoder, ctx_tokenizer, q_encoder, q_tokenizer
    """
    print("📦 Loading DPR Context Encoder …")
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )
    ctx_encoder = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    ).to(device)
    ctx_encoder.eval()

    print("📦 Loading DPR Question Encoder …")
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    q_encoder = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    ).to(device)
    q_encoder.eval()

    print("✅ DPR models loaded.\n")
    return ctx_encoder, ctx_tokenizer, q_encoder, q_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: ENCODE PASSAGES AND BUILD INDEX
# ─────────────────────────────────────────────────────────────────────────────
def encode_passages(passages: list, ctx_encoder, ctx_tokenizer, device: str = "cpu"):
    """
    Encodes all passages into dense vectors using the DPR context encoder.

    Args:
        passages: List of text passages.
        ctx_encoder: DPRContextEncoder model.
        ctx_tokenizer: Corresponding tokeniser.
        device: 'cpu' or 'cuda'.

    Returns:
        torch.Tensor of shape (num_passages, hidden_size).
    """
    embeddings = []
    with torch.no_grad():
        for passage in passages:
            inputs = ctx_tokenizer(
                passage,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(device)
            output = ctx_encoder(**inputs)
            # pooler_output is the CLS-token embedding
            embeddings.append(output.pooler_output)

    return torch.cat(embeddings, dim=0)   # (N, 768)


def encode_question(question: str, q_encoder, q_tokenizer, device: str = "cpu"):
    """
    Encodes a single question into a dense vector using the DPR question encoder.

    Returns:
        torch.Tensor of shape (1, hidden_size).
    """
    with torch.no_grad():
        inputs = q_tokenizer(
            question,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True,
        ).to(device)
        output = q_encoder(**inputs)
    return output.pooler_output   # (1, 768)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RETRIEVAL — dot product similarity
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_contexts_dot_product(
    question_embedding: torch.Tensor,
    passage_embeddings: torch.Tensor,
    passages: list,
    top_k: int = 2,
) -> list:
    """
    Retrieves the top-k most relevant passages using dot product similarity.

    Args:
        question_embedding: (1, D) tensor.
        passage_embeddings: (N, D) tensor.
        passages: List of raw passage strings.
        top_k: Number of passages to retrieve.

    Returns:
        List of the top-k passage strings.
    """
    # Dot product: (1, D) × (D, N) → (1, N)
    scores = torch.matmul(question_embedding, passage_embeddings.T).squeeze(0)
    top_indices = scores.topk(top_k).indices.tolist()
    return [passages[i] for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE: RETRIEVAL — cosine similarity  (replaces dot product)
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_contexts_cosine(
    question_embedding: torch.Tensor,
    passage_embeddings: torch.Tensor,
    passages: list,
    top_k: int = 2,
) -> list:
    """
    Retrieves the top-k most relevant passages using cosine similarity.

    Cosine similarity formula:
        cos(q, p) = (q · p) / (||q|| × ||p||)

    Unlike dot product, cosine similarity is invariant to vector magnitude,
    making it more suitable when only the direction of embeddings matters.

    Args:
        question_embedding: (1, D) tensor.
        passage_embeddings: (N, D) tensor.
        passages: List of raw passage strings.
        top_k: Number of passages to retrieve.

    Returns:
        List of the top-k passage strings (by cosine similarity).
    """
    # Step 1 — Dot product numerator
    dot_products = torch.matmul(question_embedding, passage_embeddings.T).squeeze(0)  # (N,)

    # Step 2 — Compute norms
    q_norm = question_embedding.norm(dim=-1)                  # scalar (or (1,))
    p_norms = passage_embeddings.norm(dim=-1)                 # (N,)

    # Step 3 — Cosine similarity = dot / (||q|| * ||p||)
    cosine_similarities = dot_products / (q_norm * p_norms + 1e-8)   # (N,)

    # Step 4 — Sort and select top-k
    top_indices = cosine_similarities.topk(top_k).indices.tolist()
    return [passages[i] for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: GPT-2 MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_gpt2(device: str = "cpu"):
    """
    Loads the GPT-2 language model and tokeniser.

    Returns:
        gpt2_model, gpt2_tokenizer
    """
    print("📦 Loading GPT-2 …")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token   # GPT-2 has no pad token by default

    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2_model.eval()
    print("✅ GPT-2 loaded.\n")
    return gpt2_model, gpt2_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: GENERATION WITH GPT-2
# ─────────────────────────────────────────────────────────────────────────────
def generate_answer(
    prompt: str,
    gpt2_model,
    gpt2_tokenizer,
    device: str = "cpu",
    max_length: int = 150,
    min_length: int = 30,
    length_penalty: float = 1.0,
    num_beams: int = 3,
) -> str:
    """
    Generates an answer from GPT-2 given a text prompt.

    Args:
        prompt:         The full input text (context + question).
        gpt2_model:     Loaded GPT2LMHeadModel.
        gpt2_tokenizer: Corresponding tokeniser.
        device:         'cpu' or 'cuda'.
        max_length:     Maximum total token length of generated output.
        min_length:     Minimum total token length of generated output.
        length_penalty: >1 → favour longer sequences; <1 → favour shorter.
        num_beams:      Beam search width. 1 = greedy; higher = more thorough.

    Returns:
        Generated text string (decoded).
    """
    inputs = gpt2_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = gpt2_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    generated_ids = output_ids[0][input_len:]
    return gpt2_tokenizer.decode(generated_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: COMPARE DIRECT vs. RAG GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def compare_direct_vs_rag(
    question: str,
    passage_embeddings: torch.Tensor,
    passages: list,
    ctx_encoder,
    ctx_tokenizer,
    q_encoder,
    q_tokenizer,
    gpt2_model,
    gpt2_tokenizer,
    device: str = "cpu",
    top_k: int = 2,
    use_cosine: bool = True,
):
    """
    Compares GPT-2 answers:
      A) Direct generation (no retrieval context)
      B) RAG-augmented generation (with DPR retrieved contexts)
    """
    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    # ── A: Direct generation ─────────────────────────────────────────────────
    direct_prompt = f"Question: {question}\nAnswer:"
    direct_answer = generate_answer(direct_prompt, gpt2_model, gpt2_tokenizer, device)
    print(f"\n[A] Direct GPT-2 (no context):\n{direct_answer or '[no output]'}")

    # ── B: RAG-augmented generation ───────────────────────────────────────────
    q_emb = encode_question(question, q_encoder, q_tokenizer, device)

    if use_cosine:
        retrieved = retrieve_contexts_cosine(q_emb, passage_embeddings, passages, top_k)
    else:
        retrieved = retrieve_contexts_dot_product(q_emb, passage_embeddings, passages, top_k)

    context_text = " ".join(retrieved)
    rag_prompt = f"Context: {context_text}\nQuestion: {question}\nAnswer:"
    rag_answer = generate_answer(rag_prompt, gpt2_model, gpt2_tokenizer, device)
    print(f"\n[B] RAG GPT-2 (with DPR context):\n{rag_answer or '[no output]'}")
    print(f"\nRetrieved context snippet: '{retrieved[0][:80]}…'\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: EXERCISE — Parameter Tuning
# ─────────────────────────────────────────────────────────────────────────────
def parameter_tuning_exercise(
    question: str,
    context: str,
    gpt2_model,
    gpt2_tokenizer,
    device: str = "cpu",
):
    """
    Runs the same RAG prompt through GPT-2 with three different
    generation parameter configurations and prints a comparison.

    Objective: Understand how max_length, min_length, length_penalty,
    and num_beams affect output quality, conciseness, and relevance.
    """
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    param_sets = [
        {
            "label": "Config 1 — Short & Greedy",
            "max_length": 80,
            "min_length": 10,
            "length_penalty": 0.8,   # Penalise long outputs
            "num_beams": 1,          # Greedy decoding
        },
        {
            "label": "Config 2 — Balanced Beam Search",
            "max_length": 150,
            "min_length": 30,
            "length_penalty": 1.0,   # Neutral
            "num_beams": 4,          # Moderate beam search
        },
        {
            "label": "Config 3 — Long & Exhaustive",
            "max_length": 250,
            "min_length": 60,
            "length_penalty": 1.5,   # Encourage longer output
            "num_beams": 8,          # Wide beam search
        },
    ]

    print("\n" + "=" * 70)
    print("EXERCISE: PARAMETER TUNING COMPARISON")
    print(f"Question: {question}")
    print("=" * 70)

    for cfg in param_sets:
        label = cfg.pop("label")
        answer = generate_answer(prompt, gpt2_model, gpt2_tokenizer, device, **cfg)
        print(f"\n🔧 {label}")
        print(f"   Params: {cfg}")
        print(f"   Answer: {answer or '[no output]'}")

    print(
        "\n📝 Analysis Guide:\n"
        "  • Config 1 (Greedy, short): Fastest but may cut off mid-sentence.\n"
        "  • Config 2 (Balanced beam): Good trade-off between quality and speed.\n"
        "  • Config 3 (Wide beam, long): Most thorough but slowest; may be verbose.\n"
        "  ↳ For concise factual answers, Config 2 usually performs best.\n"
        "  ↳ Increase num_beams for more coherent outputs at the cost of compute.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}\n")

    # Load models
    ctx_encoder, ctx_tokenizer, q_encoder, q_tokenizer = load_dpr_models(device)
    gpt2_model, gpt2_tokenizer = load_gpt2(device)

    # Encode all passages once
    print("🔢 Encoding passage corpus …")
    passage_embeddings = encode_passages(PASSAGES, ctx_encoder, ctx_tokenizer, device)
    print(f"✅ Encoded {len(PASSAGES)} passages → shape {tuple(passage_embeddings.shape)}\n")

    # Run comparison on two example questions
    for question in QUERY_EXAMPLES[:2]:
        compare_direct_vs_rag(
            question,
            passage_embeddings, PASSAGES,
            ctx_encoder, ctx_tokenizer,
            q_encoder, q_tokenizer,
            gpt2_model, gpt2_tokenizer,
            device,
            top_k=2,
            use_cosine=True,   # Toggle False to use dot product instead
        )

    # Run parameter tuning exercise on a specific question + its best context
    tuning_question = "How does photosynthesis work?"
    tuning_context = PASSAGES[3]   # Photosynthesis passage
    parameter_tuning_exercise(
        tuning_question, tuning_context,
        gpt2_model, gpt2_tokenizer, device
    )


if __name__ == "__main__":
    main()
