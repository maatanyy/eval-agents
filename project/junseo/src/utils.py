import json
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def _get_provider(model: str) -> str:
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    elif model.startswith("gemini"):
        return "gemini"
    else:
        raise ValueError(f"Cannot determine provider for model: {model}. Expected gpt-*, o1-*, o3-*, o4-*, or gemini-*")


def _build_openai_client():
    import openai
    base_url = os.getenv("OPENAI_BASE_URL") or None
    return openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url,
    )


def _build_gemini_client():
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai


def _openai_tools_to_gemini(openai_tools: list) -> list:
    """Convert OpenAI-format tool definitions to Gemini function_declarations."""
    declarations = []
    for tool in openai_tools:
        func = tool["function"]
        declarations.append({
            "name": func["name"],
            "description": func["description"],
            "parameters": func.get("parameters", {}),
        })
    return [{"function_declarations": declarations}]


def call_with_tools(
    model: str,
    system_prompt: str,
    user_message: str,
    tools: list,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 40,
) -> dict:
    """
    Call an LLM with tool definitions and classify whether it made a tool call.

    Returns:
        {
            "called_tool": bool,
            "tool_name": str | None,
            "tool_args": dict | None,
            "response_text": str | None,
        }
    """
    provider = _get_provider(model)

    if provider == "openai":
        return _openai_call_with_tools(model, system_prompt, user_message, tools, temperature, top_p)
    else:
        return _gemini_call_with_tools(model, system_prompt, user_message, tools, temperature, top_p, top_k)


def call_text(
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 40,
) -> str:
    """Call an LLM for a plain text response (no tools)."""
    provider = _get_provider(model)

    if provider == "openai":
        client = _build_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content

    else:
        genai = _build_gemini_client()
        gen_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
        )
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        response = gen_model.generate_content(user_message, generation_config=generation_config)
        return response.text


def _openai_call_with_tools(model, system_prompt, user_message, tools, temperature, top_p) -> dict:
    client = _build_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        top_p=top_p,
    )
    message = response.choices[0].message

    if message.tool_calls:
        tc = message.tool_calls[0]
        return {
            "called_tool": True,
            "tool_name": tc.function.name,
            "tool_args": json.loads(tc.function.arguments),
            "response_text": None,
        }
    return {
        "called_tool": False,
        "tool_name": None,
        "tool_args": None,
        "response_text": message.content,
    }


def _gemini_call_with_tools(model, system_prompt, user_message, tools, temperature, top_p, top_k) -> dict:
    genai = _build_gemini_client()
    gemini_tools = _openai_tools_to_gemini(tools)

    gen_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        tools=gemini_tools,
    )
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    response = gen_model.generate_content(user_message, generation_config=generation_config)

    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                fc = part.function_call
                return {
                    "called_tool": True,
                    "tool_name": fc.name,
                    "tool_args": dict(fc.args),
                    "response_text": None,
                }

    return {
        "called_tool": False,
        "tool_name": None,
        "tool_args": None,
        "response_text": response.text,
    }
