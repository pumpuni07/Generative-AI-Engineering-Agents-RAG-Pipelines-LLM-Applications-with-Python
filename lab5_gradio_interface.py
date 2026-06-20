"""
Lab 5: Setting Up a Gradio Interface to Interact with LLMs
===========================================================
Topic: Building interactive web front-ends for AI models using Gradio.

This file contains three progressively more complex Gradio demos:
  1. gradio_sum_demo()        — Basic numeric calculator (intro demo)
  2. gradio_sentence_combiner() — Exercise: combine two input sentences
  3. gradio_common_inputs()   — All common Gradio input types demonstrated
  4. gradio_llm_chat()        — Full LLM chatbot via IBM watsonx.ai

Learning objectives:
  - Use Gradio to build interactive front-end interfaces for LLMs
  - Implement text input fields, buttons, sliders, checkboxes, dropdowns
  - Customise and deploy web-based AI applications
  - Integrate Gradio with a backend LLM (IBM watsonx.ai Llama / Mixtral)

Key Gradio components covered:
  gr.Number, gr.Textbox, gr.Slider, gr.Dropdown, gr.CheckboxGroup,
  gr.Radio, gr.Checkbox, gr.Interface, gr.Interface.launch()

Prerequisites:
  pip install gradio==4.44.0 ibm-watsonx-ai==1.1.2 langchain==0.2.11 \\
              langchain-ibm==0.1.11 huggingface_hub==0.23.0

Environment:
  IBM watsonx.ai "skills-network" project_id is used for free access
  inside the IBM Cloud IDE. For local use, supply your own credentials.

Author notes:
  Based on IBM Skills Network lab material (Kang Wang, IBM / U. Waterloo).
  Extended with full implementations, the Exercise solution, and inline
  explanations by Jack Pumpuni Frimpong-Manso.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import gradio as gr

# IBM watsonx.ai — optional; only needed for the LLM chat demo
try:
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from langchain_ibm import WatsonxLLM
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 1: Basic Sum Calculator  (introductory Gradio demo from lab)
# ─────────────────────────────────────────────────────────────────────────────
def add_numbers(num1: float, num2: float) -> float:
    """Returns the sum of two numbers."""
    return num1 + num2


def gradio_sum_demo(server_name: str = "127.0.0.1", server_port: int = 7860):
    """
    Launches a simple numeric sum calculator using Gradio.

    Demonstrates:
      - gr.Number() as both input and output
      - gr.Interface with fn, inputs, outputs

    Run:
        python lab5_gradio_interface.py --demo sum
    """
    demo = gr.Interface(
        fn=add_numbers,
        inputs=[
            gr.Number(label="Number 1"),
            gr.Number(label="Number 2"),
        ],
        outputs=gr.Number(label="Sum"),
        title="➕ Sum Calculator",
        description="Enter two numbers and get their sum.",
    )
    demo.launch(server_name=server_name, server_port=server_port)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2 (EXERCISE): Sentence Combiner
# ─────────────────────────────────────────────────────────────────────────────
def combine_sentences(sentence1: str, sentence2: str) -> str:
    """
    Combines two input sentences into one.

    Exercise solution:
      The lab asks you to create a Gradio app that combines two sentences.
      This mirrors the sum calculator pattern but with text inputs/outputs.
    """
    s1 = sentence1.strip()
    s2 = sentence2.strip()
    if not s1 and not s2:
        return "Please enter at least one sentence."
    if not s1:
        return s2
    if not s2:
        return s1
    # Add a space between sentences; handle trailing punctuation gracefully
    separator = " " if s1.endswith((".", "!", "?", ",", ";", ":")) else ". "
    return s1 + separator + s2


def gradio_sentence_combiner(server_name: str = "127.0.0.1", server_port: int = 7860):
    """
    Launches the sentence combiner Gradio app (Exercise solution).

    Demonstrates:
      - gr.Textbox() for multi-line text input
      - Combining two string inputs into one text output

    Run:
        python lab5_gradio_interface.py --demo sentences
    """
    demo = gr.Interface(
        fn=combine_sentences,
        inputs=[
            gr.Textbox(label="Sentence 1", lines=2,
                       placeholder="Enter the first sentence here..."),
            gr.Textbox(label="Sentence 2", lines=2,
                       placeholder="Enter the second sentence here..."),
        ],
        outputs=gr.Textbox(label="Combined Output", lines=3),
        title="📝 Sentence Combiner",
        description="Type two sentences and combine them into one.",
        examples=[
            ["The sun is shining.", "It is a beautiful day."],
            ["Python is a powerful language.", "It is widely used in AI and data science."],
            ["AI is transforming industries.", "LLMs are at the forefront of this change."],
        ],
    )
    demo.launch(server_name=server_name, server_port=server_port)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3: All Common Gradio Input Types  (from lab: common_input_types.py)
# ─────────────────────────────────────────────────────────────────────────────
def sentence_builder(
    quantity: int,
    tech_worker_type: str,
    countries: list,
    place: str,
    activity_list: list,
    morning: bool,
) -> str:
    """
    Builds a descriptive sentence from multiple structured inputs.

    This function demonstrates how Gradio can collect diverse input types
    (numeric slider, dropdown, checkbox group, radio, multi-select dropdown,
    and boolean checkbox) and combine them into a coherent output.
    """
    time_of_day = "morning" if morning else "night"
    countries_str = " and ".join(countries) if countries else "somewhere"
    activities_str = " and ".join(activity_list) if activity_list else "did nothing"

    return (
        f"The {quantity} {tech_worker_type}s from {countries_str} "
        f"went to the {place} where they {activities_str} "
        f"until the {time_of_day}."
    )


def gradio_common_inputs(server_name: str = "127.0.0.1", server_port: int = 7860):
    """
    Launches a demo showcasing all common Gradio input types.

    Input types demonstrated:
      - gr.Slider       : integer range selector (Count)
      - gr.Dropdown     : single-select dropdown (tech worker type)
      - gr.CheckboxGroup: multi-select from a list (countries)
      - gr.Radio        : forced single-choice (location)
      - gr.Dropdown (multiselect=True): multi-select dropdown (activities)
      - gr.Checkbox     : boolean toggle (morning?)

    Output type:
      - "text" (shorthand for gr.Textbox)

    The examples parameter produces a pre-populated examples table below
    the interface, where each inner list is one row.

    Run:
        python lab5_gradio_interface.py --demo inputs
    """
    demo = gr.Interface(
        fn=sentence_builder,
        inputs=[
            gr.Slider(
                minimum=3,
                maximum=20,
                value=4,
                step=1,
                label="Count",
                info="Choose between 3 and 20",
            ),
            gr.Dropdown(
                choices=["Data Scientist", "Software Developer", "Software Engineer"],
                label="Tech Worker Type",
                info="Will add more tech worker types later!",
            ),
            gr.CheckboxGroup(
                choices=["Canada", "Japan", "France"],
                label="Countries",
                info="Where are they from?",
            ),
            gr.Radio(
                choices=["office", "restaurant", "meeting room"],
                label="Location",
                info="Where did they go?",
            ),
            gr.Dropdown(
                choices=["partied", "brainstormed", "coded", "fixed bugs"],
                value=["brainstormed", "fixed bugs"],
                multiselect=True,
                label="Activities",
                info="Which activities did they perform?",
            ),
            gr.Checkbox(
                label="Morning",
                info="Did they do it in the morning?",
            ),
        ],
        outputs="text",
        title="🏢 Tech Team Story Builder",
        description=(
            "Fill in the fields to generate a short story about a tech team. "
            "Demonstrates all common Gradio input types."
        ),
        examples=[
            [3,  "Software Developer", ["Canada", "Japan"],  "restaurant",    ["coded", "fixed bugs"],       True],
            [4,  "Data Scientist",     ["Japan"],            "office",         ["brainstormed", "partied"],   False],
            [10, "Software Engineer",  ["Canada", "France"], "meeting room",   ["brainstormed"],              False],
            [8,  "Data Scientist",     ["France"],           "restaurant",     ["coded"],                     True],
        ],
    )
    demo.launch(server_name=server_name, server_port=server_port)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4: Gradio + IBM watsonx.ai LLM Chat  (from lab: llm_chat.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_watsonx_llm(
    model_id: str = "meta-llama/llama-3-2-11b-vision-instruct",
    max_new_tokens: int = 256,
    temperature: float = 0.5,
    project_id: str = "skills-network",
):
    """
    Initialises a WatsonxLLM instance from IBM watsonx.ai.

    Model options:
      - 'meta-llama/llama-3-2-11b-vision-instruct'        (Llama — default)
      - 'mistralai/mistral-small-3-1-24b-instruct-2503'   (Mixtral)
      - 'ibm/granite-4-h-small'                           (Granite)

    To switch models, change the model_id parameter.

    Note:
      project_id="skills-network" gives free access inside IBM Cloud IDE.
      For local use, replace with your own IBM watsonx.ai project ID and
      set up credentials via ibm_watsonx_ai.Credentials.

    Args:
        model_id:       Watsonx foundation model identifier.
        max_new_tokens: Maximum tokens to generate per response.
                        Increase this if responses are truncated.
        temperature:    Sampling temperature (0 = deterministic, 1 = creative).
        project_id:     IBM watsonx.ai project ID.

    Returns:
        WatsonxLLM instance ready for .invoke() calls.
    """
    if not WATSONX_AVAILABLE:
        raise ImportError(
            "IBM watsonx.ai packages not installed.\n"
            "Run: pip install ibm-watsonx-ai langchain-ibm"
        )

    parameters = {
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.TEMPERATURE: temperature,
    }

    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm


def gradio_llm_chat(
    model_id: str = "meta-llama/llama-3-2-11b-vision-instruct",
    max_new_tokens: int = 256,
    temperature: float = 0.5,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
):
    """
    Launches a Gradio chatbot backed by IBM watsonx.ai.

    Exercise solution note:
      If responses are incomplete, increase max_new_tokens (e.g. 512 or 1024).
      This is the most common cause of truncated LLM output in Gradio apps.

    Run:
        python lab5_gradio_interface.py --demo chat
    """
    watsonx_llm = build_watsonx_llm(model_id, max_new_tokens, temperature)

    def generate_response(prompt_txt: str) -> str:
        """Calls the LLM and returns the generated text."""
        if not prompt_txt.strip():
            return "Please enter a question."
        return watsonx_llm.invoke(prompt_txt)

    chat_application = gr.Interface(
        fn=generate_response,
        allow_flagging="never",
        inputs=gr.Textbox(
            label="Input",
            lines=2,
            placeholder="Type your question here...",
        ),
        outputs=gr.Textbox(label="Output"),
        title="🤖 Watsonx.ai Chatbot",
        description=(
            f"Ask any question and the chatbot will answer using {model_id}.\n"
            f"Model: {model_id} | Temperature: {temperature} | Max tokens: {max_new_tokens}"
        ),
        examples=[
            ["How do I become a good data scientist?"],
            ["Explain large language models in simple terms."],
            ["What are the benefits of retrieval-augmented generation?"],
        ],
    )
    chat_application.launch(server_name=server_name, server_port=server_port)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
DEMO_MAP = {
    "sum":       gradio_sum_demo,
    "sentences": gradio_sentence_combiner,
    "inputs":    gradio_common_inputs,
    "chat":      gradio_llm_chat,
}

if __name__ == "__main__":
    import sys

    demo_choice = "inputs"   # Default demo to run
    if "--demo" in sys.argv:
        idx = sys.argv.index("--demo")
        if idx + 1 < len(sys.argv):
            demo_choice = sys.argv[idx + 1]

    if demo_choice not in DEMO_MAP:
        print(f"Unknown demo '{demo_choice}'. Available: {list(DEMO_MAP.keys())}")
        sys.exit(1)

    print(f"\n🚀 Launching Gradio demo: '{demo_choice}'")
    print("   Open http://127.0.0.1:7860 in your browser.\n")
    DEMO_MAP[demo_choice]()
