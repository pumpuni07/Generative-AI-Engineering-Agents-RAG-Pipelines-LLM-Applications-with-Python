"""
Lab 6: IBM watsonx.ai LLM Quickstart
======================================
Topic: Using IBM watsonx.ai's API to create a Q&A bot with Llama and Mixtral.

This lab covers:
  1. Connecting to IBM watsonx.ai via the Python SDK + LangChain wrapper
  2. Choosing between Llama, Mixtral, and Granite foundation models
  3. Configuring generation parameters (MAX_NEW_TOKENS, TEMPERATURE)
  4. Running a simple terminal Q&A bot
  5. Key differences between model families

Supported use cases for Llama / Mixtral:
  - Questions & Answers (Q&A)
  - Summarization
  - Classification
  - Generation
  - Extraction
  - Retrieval-Augmented Generation (RAG)
  - Code generation

Additional Granite-only use cases:
  - Reasoning & planning
  - Fill-in-the-middle

Note on access:
  project_id="skills-network" provides free access inside IBM Cloud IDE.
  For local environments, use your own IBM Cloud API key and project ID.
  See: https://ibm.github.io/watsonx-ai-python-sdk/

Prerequisites:
  pip install ibm-watsonx-ai==1.1.2 langchain-ibm==0.1.11 langchain==0.2.11

Author notes:
  Based on IBM Skills Network lab material (Kang Wang, IBM / U. Waterloo).
  Extended with model comparison, parameter explanations, and exercises
  by Jack Pumpuni Frimpong-Manso.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

try:
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from langchain_ibm import WatsonxLLM
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False
    print(
        "⚠️  IBM watsonx.ai packages not found.\n"
        "   Install with: pip install ibm-watsonx-ai langchain-ibm\n"
        "   Running in DEMO MODE — LLM calls are simulated.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "llama":   "meta-llama/llama-3-2-11b-vision-instruct",
    "mixtral": "mistralai/mistral-small-3-1-24b-instruct-2503",
    "granite": "ibm/granite-4-h-small",
}

# Default: Llama (as per lab)
DEFAULT_MODEL_KEY = "llama"

WATSONX_URL      = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID       = "skills-network"   # Free access in IBM Cloud IDE


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: INITIALISE THE LLM
# ─────────────────────────────────────────────────────────────────────────────
def get_llm(
    model_key: str = DEFAULT_MODEL_KEY,
    max_new_tokens: int = 256,
    temperature: float = 0.5,
) -> "WatsonxLLM":
    """
    Initialises and returns a WatsonxLLM instance.

    Args:
        model_key:      One of 'llama', 'mixtral', 'granite'.
        max_new_tokens: Maximum tokens in the generated response.
                        Increase if answers are truncated (e.g. 512).
        temperature:    Controls randomness (0=deterministic, 1=very creative).
                        0.5 balances accuracy and variety for Q&A tasks.

    Returns:
        WatsonxLLM: A LangChain-compatible LLM wrapper around the IBM model.

    Example:
        llm = get_llm("mixtral", max_new_tokens=512, temperature=0.3)
        response = llm.invoke("What is retrieval-augmented generation?")
    """
    if not WATSONX_AVAILABLE:
        raise RuntimeError("IBM watsonx.ai not installed. See module docstring.")

    model_id = AVAILABLE_MODELS.get(model_key)
    if not model_id:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Choose from: {list(AVAILABLE_MODELS.keys())}"
        )

    parameters = {
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.TEMPERATURE:    temperature,
    }

    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=parameters,
    )

    print(f"✅ Loaded model: {model_id}")
    return watsonx_llm


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SIMPLE TERMINAL Q&A BOT
# ─────────────────────────────────────────────────────────────────────────────
def run_simple_qa(model_key: str = DEFAULT_MODEL_KEY):
    """
    Runs a single-turn Q&A bot in the terminal.

    The user types a query; the LLM generates and prints a response.
    This is the core pattern from simple_llm.py in the lab.

    Usage:
        python lab6_watsonx_llm_quickstart.py --mode single
    """
    print(f"\n{'─'*60}")
    print(f" IBM watsonx.ai Q&A Bot  |  Model: {model_key.upper()}")
    print(f"{'─'*60}\n")

    llm = get_llm(model_key)

    # Get query from user (mirrors lab's input() call)
    query = input("Please enter your query: ").strip()
    if not query:
        print("No query entered.")
        return

    print("\n⏳ Generating response …\n")
    response = llm.invoke(query)
    print(f"Response:\n{response}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def compare_models(query: str, max_new_tokens: int = 256, temperature: float = 0.5):
    """
    Runs the same query through Llama, Mixtral, and Granite and prints
    a side-by-side comparison.

    Args:
        query:          The question to ask all models.
        max_new_tokens: Token limit per model.
        temperature:    Sampling temperature.

    Usage:
        python lab6_watsonx_llm_quickstart.py --mode compare
    """
    print(f"\n{'='*60}")
    print(f" MODEL COMPARISON")
    print(f" Query: {query}")
    print(f"{'='*60}\n")

    for key, model_id in AVAILABLE_MODELS.items():
        print(f"── {key.upper()} ({model_id}) ──")
        try:
            llm = get_llm(key, max_new_tokens, temperature)
            response = llm.invoke(query)
            print(f"{response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: INTERACTIVE CHAT LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_interactive_chat(model_key: str = DEFAULT_MODEL_KEY):
    """
    Runs a multi-turn interactive Q&A session in the terminal.
    Type 'exit', 'quit', or 'bye' to end.

    NOTE: watsonx.ai models are stateless by default — each query is
    independent. For multi-turn memory, use LangChain ConversationChain
    (see Lab 4 for an example with conversation history).

    Usage:
        python lab6_watsonx_llm_quickstart.py --mode chat
    """
    print(f"\n{'─'*60}")
    print(f" IBM watsonx.ai Interactive Chat  |  Model: {model_key.upper()}")
    print(f" Type 'exit' to quit.")
    print(f"{'─'*60}\n")

    llm = get_llm(model_key)

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit", "bye"}:
            print("Bot: Goodbye! Have a great day.")
            break

        try:
            response = llm.invoke(query)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Bot: [Error — {e}]\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: PARAMETER TUNING NOTES  (Exercise reference)
# ─────────────────────────────────────────────────────────────────────────────
"""
Exercise: Fix incomplete responses
───────────────────────────────────
Problem:  The LLM sometimes generates incomplete answers.
Cause:    MAX_NEW_TOKENS is too low — the model hits the token limit
          before finishing its response.
Solution: Increase MAX_NEW_TOKENS:

    parameters = {
        GenParams.MAX_NEW_TOKENS: 512,   # was 256 — doubled
        GenParams.TEMPERATURE:    0.5,
    }

Parameter guide:
  MAX_NEW_TOKENS | Effect
  ─────────────────────────────────────────────────────────────
  64–128         | Very short answers; likely to truncate.
  256            | Lab default; adequate for concise responses.
  512            | Recommended for detailed explanations.
  1024+          | Long-form content (summaries, essays).

  TEMPERATURE    | Effect
  ─────────────────────────────────────────────────────────────
  0.0–0.2        | Near-deterministic; best for factual Q&A.
  0.5            | Balanced (lab default).
  0.8–1.0        | Creative; more varied but may hallucinate.
  >1.0           | Experimental; often incoherent.

Model selection guide:
  Model     | Best for
  ──────────────────────────────────────────────────────────────
  Llama     | General Q&A, code generation, multilingual tasks
  Mixtral   | Longer context, reasoning, summarisation
  Granite   | IBM-specific tasks, reasoning, fill-in-the-middle
"""


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    mode       = "single"
    model_key  = DEFAULT_MODEL_KEY

    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]

    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_key = sys.argv[idx + 1]

    if not WATSONX_AVAILABLE:
        print(
            "Demo mode (no IBM packages installed).\n"
            "Available modes: --mode single | chat | compare\n"
            "Available models: --model llama | mixtral | granite\n"
        )
        sys.exit(0)

    if mode == "single":
        run_simple_qa(model_key)
    elif mode == "chat":
        run_interactive_chat(model_key)
    elif mode == "compare":
        query = input("Enter query to compare across models: ").strip()
        compare_models(query)
    else:
        print(f"Unknown mode '{mode}'. Use: single | chat | compare")
