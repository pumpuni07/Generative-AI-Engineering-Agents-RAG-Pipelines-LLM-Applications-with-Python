"""
Lab 3: RAG with PyTorch — Song Content Appropriateness for Children
====================================================================
Topic: Building a RAG pipeline in pure PyTorch (no Hugging Face pipeline layer)
       to classify whether song content is appropriate for children.

Business context (from lab):
  A social media company needs to determine whether songs shared on its platform
  are child-appropriate. Running a full LLM on every song is cost-prohibitive.
  RAG offers a scalable alternative: embed pre-answered questions about content
  appropriateness, retrieve the most similar Q&A pair for a new song snippet,
  and return the pre-vetted answer.

This lab covers:
  1. Creating sentence embeddings with a lightweight transformer (MiniLM).
  2. Implementing RAG_QA() with DOT PRODUCT similarity.
  3. Exercise: Replacing dot product with COSINE SIMILARITY and explaining why.
  4. Evaluation of both retrieval methods on a test set.

Prerequisites:
  pip install torch sentence-transformers

Author notes:
  Based on IBM Skills Network lab material.
  Extended with complete working implementation, cosine exercise solution,
  and comparative evaluation.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE: Pre-answered Content Appropriateness Q&A pairs
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is (question, answer). The questions describe song characteristics;
# the answers give the appropriateness verdict with reasoning.
QA_PAIRS: List[Tuple[str, str]] = [
    (
        "Does the song contain explicit violence or references to hurting people?",
        "NOT APPROPRIATE. Songs with explicit violence or harm towards others "
        "are not suitable for children as they can normalise aggressive behaviour."
    ),
    (
        "Does the song include profanity or offensive language?",
        "NOT APPROPRIATE. Profanity and offensive language are not suitable for "
        "children and violate community guidelines for child audiences."
    ),
    (
        "Does the song discuss romantic love in a wholesome, age-appropriate way?",
        "APPROPRIATE. Wholesome romantic themes, such as friendship and kindness, "
        "are perfectly suitable for children."
    ),
    (
        "Does the song contain explicit sexual content or adult themes?",
        "NOT APPROPRIATE. Explicit sexual content is strictly unsuitable for "
        "children and must be flagged for age-gating."
    ),
    (
        "Does the song promote positive values like friendship, kindness, or teamwork?",
        "APPROPRIATE. Songs promoting positive social values are excellent for "
        "children and encouraged on the platform."
    ),
    (
        "Does the song reference drug use or substance abuse?",
        "NOT APPROPRIATE. References to drug or alcohol use are harmful to "
        "children and should not be accessible to young audiences."
    ),
    (
        "Is the song a fun, upbeat track about animals or nature?",
        "APPROPRIATE. Educational and fun content about animals and nature is "
        "highly suitable for children of all ages."
    ),
    (
        "Does the song include hate speech or discriminatory language targeting groups?",
        "NOT APPROPRIATE. Hate speech and discriminatory content violate "
        "platform policies and are entirely unsuitable for children."
    ),
    (
        "Does the song encourage children to be brave, curious, or creative?",
        "APPROPRIATE. Songs that inspire curiosity, bravery, and creativity "
        "are beneficial and child-friendly."
    ),
    (
        "Does the song depict or glorify criminal activity such as theft or assault?",
        "NOT APPROPRIATE. Glorifying criminal activity is harmful to children "
        "and is flagged for removal from child-accessible playlists."
    ),
]

# Test song snippets to evaluate (new songs not in the knowledge base)
TEST_SONGS: List[Tuple[str, str]] = [
    ("Let's go on a rainbow adventure with our animal friends!", "APPROPRIATE"),
    ("I'll hurt anyone who gets in my way, no regrets!", "NOT APPROPRIATE"),
    ("Be kind, share your toys, and make a friend today.", "APPROPRIATE"),
    ("Sipping on drinks all night, living the wild life.", "NOT APPROPRIATE"),
    ("Together we rise, we build, we dream and we inspire.", "APPROPRIATE"),
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads a lightweight sentence embedding model.
    MiniLM-L6-v2 produces 384-dimensional embeddings and is fast on CPU.
    """
    print(f"📦 Loading embedding model: {model_name} …")
    model = SentenceTransformer(model_name)
    print("✅ Embedding model loaded.\n")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: EMBED THE KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
def embed_qa_questions(
    qa_pairs: List[Tuple[str, str]],
    model: SentenceTransformer,
) -> torch.Tensor:
    """
    Encodes only the questions from the Q&A pairs into dense vectors.

    The intuition: at retrieval time, we match a song snippet against
    the *questions* to find the most semantically relevant pre-answered
    content-appropriateness query.

    Returns:
        torch.Tensor of shape (num_qa_pairs, embedding_dim).
    """
    questions = [q for q, _ in qa_pairs]
    embeddings = model.encode(questions, convert_to_tensor=True)
    return embeddings   # (N, D)


def embed_text(text: str, model: SentenceTransformer) -> torch.Tensor:
    """
    Encodes a single text string (e.g., a song snippet) into a dense vector.

    Returns:
        torch.Tensor of shape (1, embedding_dim).
    """
    embedding = model.encode([text], convert_to_tensor=True)
    return embedding   # (1, D)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RAG_QA — Dot Product Version (original)
# ─────────────────────────────────────────────────────────────────────────────
def RAG_QA_dot_product(
    embeddings_questions: torch.Tensor,
    embedding_query: torch.Tensor,
    qa_pairs: List[Tuple[str, str]],
    top_k: int = 1,
) -> List[Tuple[str, str, float]]:
    """
    Retrieves the most relevant Q&A pairs using DOT PRODUCT similarity.

    The dot product captures both magnitude and direction of the embedding
    vectors. It is equivalent to cosine similarity when embeddings are
    L2-normalised (as in many SentenceTransformer models), but differs
    when they are not.

    Args:
        embeddings_questions: (N, D) tensor of encoded Q&A questions.
        embedding_query:      (1, D) tensor of the encoded song snippet.
        qa_pairs:             List of (question, answer) tuples.
        top_k:                Number of top matches to retrieve.

    Returns:
        List of (question, answer, score) tuples, sorted by relevance.
    """
    # (1, D) × (D, N) → (1, N) → squeeze → (N,)
    scores = torch.matmul(embedding_query, embeddings_questions.T).squeeze(0)

    top_indices = scores.topk(top_k).indices.tolist()
    return [
        (qa_pairs[i][0], qa_pairs[i][1], scores[i].item())
        for i in top_indices
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: RAG_QA — Cosine Similarity Version (EXERCISE SOLUTION)
# ─────────────────────────────────────────────────────────────────────────────
def RAG_QA(
    embeddings_questions: torch.Tensor,
    embedding_query: torch.Tensor,
    qa_pairs: List[Tuple[str, str]],
    top_k: int = 1,
) -> List[Tuple[str, str, float]]:
    """
    Retrieves the most relevant Q&A pairs using COSINE SIMILARITY.

    ─── Exercise Solution ────────────────────────────────────────────────────
    Why cosine similarity instead of dot product?

    Cosine similarity measures ONLY the angle (direction) between two vectors,
    ignoring their magnitudes:

        cos(q, p) = (q · p) / (||q|| × ||p||)

    This is important in text/embedding tasks because:
      • The magnitude of an embedding can reflect token count or other
        artefacts unrelated to semantic meaning.
      • Two documents with identical semantic content but different lengths
        may have different embedding magnitudes.
      • Cosine similarity gives values in [-1, 1], making scores comparable
        across different queries and knowledge bases.

    When embeddings ARE unit-normalised (as in many SentenceTransformer
    checkpoints), cosine similarity and dot product give the same ranking —
    but cosine is safer because it is magnitude-invariant by definition.
    ──────────────────────────────────────────────────────────────────────────

    Args:
        embeddings_questions: (N, D) tensor of encoded Q&A questions.
        embedding_query:      (1, D) tensor of the encoded song snippet.
        qa_pairs:             List of (question, answer) tuples.
        top_k:                Number of top matches to retrieve.

    Returns:
        List of (question, answer, score) tuples, sorted by relevance.
    """
    # Step 1 — Dot product numerator: (N,)
    dot_products = torch.matmul(embedding_query, embeddings_questions.T).squeeze(0)

    # Step 2 — Norms
    q_norm = embedding_query.norm(dim=-1)          # scalar
    p_norms = embeddings_questions.norm(dim=-1)    # (N,)

    # Step 3 — Cosine similarity
    cosine_scores = dot_products / (q_norm * p_norms + 1e-8)   # (N,)

    # Step 4 — Sort and select top-k
    top_indices = cosine_scores.topk(top_k).indices.tolist()
    return [
        (qa_pairs[i][0], qa_pairs[i][1], cosine_scores[i].item())
        for i in top_indices
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    test_songs: List[Tuple[str, str]],
    embeddings_questions: torch.Tensor,
    qa_pairs: List[Tuple[str, str]],
    model: SentenceTransformer,
    use_cosine: bool = True,
    top_k: int = 1,
):
    """
    Evaluates the RAG pipeline on test song snippets.

    Args:
        test_songs:    List of (snippet, expected_label) tuples.
        use_cosine:    If True, use cosine similarity; else use dot product.
        top_k:         Number of results to retrieve.
    """
    method = "Cosine Similarity" if use_cosine else "Dot Product"
    print(f"\n{'='*70}")
    print(f"EVALUATION — Retrieval Method: {method}")
    print(f"{'='*70}\n")

    correct = 0
    for snippet, expected in test_songs:
        emb_snippet = embed_text(snippet, model)

        if use_cosine:
            results = RAG_QA(embeddings_questions, emb_snippet, qa_pairs, top_k)
        else:
            results = RAG_QA_dot_product(embeddings_questions, emb_snippet, qa_pairs, top_k)

        top_question, top_answer, top_score = results[0]
        predicted = "APPROPRIATE" if "APPROPRIATE" in top_answer and "NOT" not in top_answer[:4] else "NOT APPROPRIATE"
        # More robust label extraction
        predicted = "NOT APPROPRIATE" if top_answer.startswith("NOT APPROPRIATE") else "APPROPRIATE"

        match = predicted == expected
        if match:
            correct += 1

        status = "✅" if match else "❌"
        print(f"{status} Snippet : {snippet[:60]}…" if len(snippet) > 60 else f"{status} Snippet : {snippet}")
        print(f"   Expected    : {expected}")
        print(f"   Predicted   : {predicted}")
        print(f"   Matched Q&A : {top_question[:60]}…" if len(top_question) > 60 else f"   Matched Q&A : {top_question}")
        print(f"   Score       : {top_score:.4f}\n")

    accuracy = correct / len(test_songs) * 100
    print(f"📊 Accuracy ({method}): {correct}/{len(test_songs)} = {accuracy:.1f}%\n")
    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: SINGLE SONG DEMO
# ─────────────────────────────────────────────────────────────────────────────
def check_song(
    song_snippet: str,
    embeddings_questions: torch.Tensor,
    qa_pairs: List[Tuple[str, str]],
    model: SentenceTransformer,
) -> str:
    """
    Checks whether a single song snippet is appropriate for children.

    Returns:
        "APPROPRIATE" or "NOT APPROPRIATE"
    """
    emb = embed_text(song_snippet, model)
    results = RAG_QA(embeddings_questions, emb, qa_pairs, top_k=1)
    _, answer, score = results[0]
    label = "NOT APPROPRIATE" if answer.startswith("NOT APPROPRIATE") else "APPROPRIATE"
    print(f"\n🎵 Song snippet : '{song_snippet}'")
    print(f"   Verdict      : {label}")
    print(f"   Confidence   : {score:.4f}")
    print(f"   Reasoning    : {answer}\n")
    return label


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Load model
    model = load_embedding_model()

    # Embed the Q&A knowledge base (questions only)
    print("🔢 Embedding knowledge base …")
    embeddings_questions = embed_qa_questions(QA_PAIRS, model)
    print(f"✅ Knowledge base embedded → shape {tuple(embeddings_questions.shape)}\n")

    # ── Demo: single song check ───────────────────────────────────────────────
    demo_songs = [
        "Let's dance and sing under the rainbow with butterflies!",
        "I'm going to destroy everything and take what I want.",
        "Friends forever, sharing and caring every day.",
    ]
    print("── Single Song Checks ──────────────────────────────────────────────")
    for song in demo_songs:
        check_song(song, embeddings_questions, QA_PAIRS, model)

    # ── Evaluation: dot product vs cosine ─────────────────────────────────────
    acc_dot = evaluate(TEST_SONGS, embeddings_questions, QA_PAIRS, model, use_cosine=False)
    acc_cos = evaluate(TEST_SONGS, embeddings_questions, QA_PAIRS, model, use_cosine=True)

    print(f"\n📈 Summary:")
    print(f"   Dot Product Accuracy    : {acc_dot:.1f}%")
    print(f"   Cosine Similarity Acc.  : {acc_cos:.1f}%")
    print(
        "\n💡 Key insight:\n"
        "   Cosine similarity is preferred for text embeddings because it\n"
        "   measures semantic direction regardless of vector magnitude, making\n"
        "   it more robust when comparing snippets of different lengths.\n"
    )


if __name__ == "__main__":
    main()
