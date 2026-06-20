"""
Lab 4: In-Context Engineering and Prompt Templates with LangChain
=================================================================
Topic: Controlling LLM behaviour through prompt engineering techniques:
       system roles, tone injection, few-shot examples, and LangChain chains.

This lab demonstrates:
  1. Role-based prompt templates ("You are a game master …")
  2. Tone injection ("respond in a formal/casual/pirate tone")
  3. Few-shot prompting for consistent output format
  4. Building an LLMChain with LangChain's PromptTemplate + ChatOpenAI
  5. Exercise 1: Changing LLM parameters (temperature, max_tokens, model)
  6. Interactive chat loop that maintains conversation context

Key insight (from lab):
  In-context engineering allows you to radically change LLM output style
  and persona without any fine-tuning — just by carefully crafting the
  prompt structure.

Prerequisites:
  pip install langchain langchain-openai openai python-dotenv

Environment variables:
  OPENAI_API_KEY=<your_key>    # Required for OpenAI models
  # Or use a local model via Ollama (see OLLAMA section below)

Author notes:
  Based on IBM Skills Network lab material.
  Extended with full working implementations, Exercise 1 solution,
  few-shot prompting, and conversation memory.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# LangChain v0.2+ imports
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Model imports (OpenAI)
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Optional: Ollama for local models
try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL   = "gpt-3.5-turbo"
DEFAULT_TEMP    = 0.7
DEFAULT_TOKENS  = 512


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = DEFAULT_TOKENS,
    use_ollama: bool = False,
    ollama_model: str = "llama3",
):
    """
    Returns an LLM instance.

    Priority:
      1. Ollama local model (if use_ollama=True and available)
      2. OpenAI ChatOpenAI (if API key is set)
      3. Raises RuntimeError with setup instructions.

    Args:
        model:        OpenAI model name (e.g. 'gpt-3.5-turbo', 'gpt-4').
        temperature:  Sampling temperature. 0=deterministic, 1=creative.
        max_tokens:   Maximum tokens in the response.
        use_ollama:   Use local Ollama model instead of OpenAI.
        ollama_model: Name of the local Ollama model to use.

    Returns:
        LangChain LLM / Chat model instance.
    """
    if use_ollama and OLLAMA_AVAILABLE:
        print(f"🦙 Using Ollama local model: {ollama_model}")
        return Ollama(model=ollama_model, temperature=temperature)

    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        print(f"🤖 Using OpenAI model: {model} (temp={temperature}, max_tokens={max_tokens})")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=OPENAI_API_KEY,
        )

    raise RuntimeError(
        "No LLM available.\n"
        "  Option A: Set OPENAI_API_KEY in your .env file and install:\n"
        "    pip install langchain-openai openai\n"
        "  Option B: Install Ollama (https://ollama.ai) and run:\n"
        "    ollama pull llama3\n"
        "    pip install langchain-community\n"
        "  Then call get_llm(use_ollama=True)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

# ── 2a. Role + Tone Template (from lab) ──────────────────────────────────────
ROLE_TONE_TEMPLATE = PromptTemplate(
    input_variables=["role", "tone", "question"],
    template=(
        "You are a {role}. "
        "Respond in a {tone} tone.\n\n"
        "User: {question}\n"
        "Answer:"
    ),
)

# ── 2b. Game Master Template (lab example) ───────────────────────────────────
GAME_MASTER_TEMPLATE = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a game master running a fantasy tabletop RPG campaign. "
        "You are creative, dramatic, and always keep the player engaged. "
        "Describe the world vividly and give the player meaningful choices.\n\n"
        "Player: {question}\n"
        "Game Master:"
    ),
)

# ── 2c. Few-Shot Template ─────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris. 🗼",
    },
    {
        "question": "Who invented the telephone?",
        "answer": "The telephone was invented by Alexander Graham Bell in 1876. 📞",
    },
    {
        "question": "What is the largest planet in the solar system?",
        "answer": "Jupiter is the largest planet in our solar system. 🪐",
    },
]

FEW_SHOT_EXAMPLE_TEMPLATE = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}",
)

FEW_SHOT_PROMPT = FewShotPromptTemplate(
    examples=FEW_SHOT_EXAMPLES,
    example_prompt=FEW_SHOT_EXAMPLE_TEMPLATE,
    prefix=(
        "You are a friendly, concise knowledge assistant. "
        "Always end your answer with a relevant emoji."
    ),
    suffix="Question: {question}\nAnswer:",
    input_variables=["question"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: LLM CHAINS
# ─────────────────────────────────────────────────────────────────────────────
def build_role_tone_chain(llm):
    """
    Builds a chain: RoleTonePromptTemplate → LLM → StrOutputParser.
    Returns the chain and a callable invoke wrapper.
    """
    chain = ROLE_TONE_TEMPLATE | llm | StrOutputParser()
    return chain


def build_game_master_chain(llm):
    """
    Builds a chain for the Game Master persona.
    """
    chain = GAME_MASTER_TEMPLATE | llm | StrOutputParser()
    return chain


def build_few_shot_chain(llm):
    """
    Builds a chain that uses few-shot examples to guide output format.
    """
    chain = FEW_SHOT_PROMPT | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: EXERCISE 1 — Change LLM Parameters
# ─────────────────────────────────────────────────────────────────────────────
def exercise_1_parameter_comparison(llm_factory_kwargs: dict = None):
    """
    Exercise 1: Change parameters for the LLM and compare outputs.

    Demonstrates how temperature and max_tokens affect response creativity
    and length on the same role-tone prompt.

    Parameter configurations tested:
      A) Low temperature (0.1)  → deterministic, factual
      B) High temperature (1.2) → creative, sometimes unpredictable
      C) Low max_tokens (50)    → brief answer, may truncate
      D) High max_tokens (300)  → detailed answer

    NOTE: This function prints comparison outputs.
    Actual LLM calls are made if a valid API key / local model is available.
    """
    test_question = "Explain photosynthesis to a 10-year-old."
    role = "science teacher"

    configs = [
        {"label": "Low Temp (0.1) — Deterministic",  "temperature": 0.1, "max_tokens": 150},
        {"label": "High Temp (1.2) — Creative",       "temperature": 1.2, "max_tokens": 150},
        {"label": "Low max_tokens (50) — Brief",      "temperature": 0.7, "max_tokens": 50},
        {"label": "High max_tokens (400) — Detailed", "temperature": 0.7, "max_tokens": 400},
    ]

    print("\n" + "=" * 70)
    print("EXERCISE 1: LLM PARAMETER COMPARISON")
    print(f"Question: {test_question}")
    print("=" * 70)

    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n🔧 {label} | Params: {cfg}")
        try:
            llm = get_llm(**cfg)
            chain = build_role_tone_chain(llm)
            response = chain.invoke({"role": role, "tone": "friendly", "question": test_question})
            print(f"   Response: {response[:300]}{'…' if len(response) > 300 else ''}")
        except RuntimeError as e:
            print(f"   ⚠️  LLM not available: {e}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
        finally:
            cfg["label"] = label   # Restore for re-use

    print(
        "\n📝 Analysis:\n"
        "  • temperature=0.1 : Consistent, factual, slightly dry.\n"
        "  • temperature=1.2 : More varied and creative; may drift off-topic.\n"
        "  • max_tokens=50   : Answer truncated — adequate only for very short responses.\n"
        "  • max_tokens=400  : Full, detailed explanation with room for examples.\n"
        "  ↳ For educational explanations, temperature ≈ 0.5–0.7 + generous max_tokens works best.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: CONVERSATION MEMORY (manual history management)
# ─────────────────────────────────────────────────────────────────────────────
class ConversationManager:
    """
    Manages multi-turn conversation history for a role-playing chatbot.
    Uses LangChain's ChatPromptTemplate with explicit message history.
    """

    def __init__(self, llm, role: str, tone: str):
        self.llm = llm
        self.role = role
        self.tone = tone
        self.history: list = []   # List of (HumanMessage, AIMessage) pairs

        # Build the system prompt
        self.system_prompt = (
            f"You are a {role}. "
            f"Respond in a {tone} tone. "
            "Be consistent with your persona across the entire conversation."
        )

    def chat(self, user_message: str) -> str:
        """
        Sends a user message and returns the assistant's reply.
        Maintains full conversation history.
        """
        # Build message list: system + history + new user message
        messages = [SystemMessage(content=self.system_prompt)]
        for human_msg, ai_msg in self.history:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))
        messages.append(HumanMessage(content=user_message))

        response = self.llm.invoke(messages)
        ai_reply = response.content if hasattr(response, "content") else str(response)

        # Store in history
        self.history.append((user_message, ai_reply))
        return ai_reply

    def clear_history(self):
        self.history = []
        print("🧹 Conversation history cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: INTERACTIVE CHAT LOOP (from lab)
# ─────────────────────────────────────────────────────────────────────────────
def run_game_master_chat(llm):
    """
    Interactive Game Master chatbot loop.

    Reproduces the lab's while-loop pattern with the role/tone template.
    Type 'quit', 'exit', or 'bye' to end.

    Test it by asking: "Who are you?"
    Expected: "I am a game master …"
    """
    role  = "game master"
    tone  = "dramatic and immersive"

    print("\n🎲 Welcome to the Fantasy RPG Chatbot!")
    print(f"Role: {role} | Tone: {tone}")
    print("Test it by asking: 'Who are you?'")
    print("Type 'quit', 'exit', or 'bye' to end.\n")

    manager = ConversationManager(llm, role, tone)

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("Answer: Farewell, brave adventurer!")
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit", "bye"}:
            print("Answer: Goodbye! May your travels be safe, adventurer.")
            break

        try:
            response = manager.chat(query)
            print(f"Answer: {response}\n")
        except Exception as e:
            print(f"Answer: [Error — {e}]\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: DEMO — All Prompt Strategies
# ─────────────────────────────────────────────────────────────────────────────
def run_demo(llm):
    """Demonstrates all three prompt strategies on a sample question."""
    question = "What is artificial intelligence?"

    print("\n" + "=" * 70)
    print("DEMO: IN-CONTEXT ENGINEERING STRATEGIES")
    print(f"Question: {question}")
    print("=" * 70)

    # Strategy A: Role + Tone
    chain_a = build_role_tone_chain(llm)
    resp_a = chain_a.invoke({"role": "professor", "tone": "formal academic", "question": question})
    print(f"\n[A] Role=Professor, Tone=Formal:\n{resp_a}")

    chain_b = build_role_tone_chain(llm)
    resp_b = chain_b.invoke({"role": "pirate", "tone": "pirate slang", "question": question})
    print(f"\n[B] Role=Pirate, Tone=Pirate Slang:\n{resp_b}")

    # Strategy B: Few-Shot
    chain_c = build_few_shot_chain(llm)
    resp_c = chain_c.invoke({"question": question})
    print(f"\n[C] Few-Shot (consistent format with emoji):\n{resp_c}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import sys

    print("─" * 70)
    print("Lab 4: In-Context Engineering & Prompt Templates")
    print("─" * 70)

    # Attempt to load LLM
    try:
        llm = get_llm(temperature=DEFAULT_TEMP, max_tokens=DEFAULT_TOKENS)
    except RuntimeError as e:
        print(f"\n⚠️  {e}")
        print("\n💡 Running in DEMO MODE (no LLM calls — showing prompts only).\n")
        # Show what the prompts look like without calling the LLM
        print("Role+Tone Prompt example:")
        print(ROLE_TONE_TEMPLATE.format(role="game master", tone="dramatic", question="Who are you?"))
        print("\nFew-Shot Prompt example:")
        print(FEW_SHOT_PROMPT.format(question="What is the boiling point of water?"))
        return

    # Run demo
    run_demo(llm)

    # Exercise 1
    exercise_1_parameter_comparison()

    # Interactive chat
    if "--chat" in sys.argv:
        run_game_master_chat(llm)
    else:
        print(
            "\n💬 To start the interactive Game Master chat, run:\n"
            "   python lab4_langchain_prompt_engineering.py --chat\n"
        )


if __name__ == "__main__":
    main()
