def load_prompt(markdown_path: str) -> str:
    """
    Loads a prompt from a given Markdown file path.

    Args:
        markdown_path (str): Path to the .md file containing the prompt.

    Returns:
        str: The prompt text as a single string.
    """
    try:
        with open(markdown_path, "r", encoding="utf-8") as file:
            prompt = file.read().strip()
        return prompt
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File not found: {markdown_path}")
    except Exception as e:
        raise RuntimeError(f"⚠️ Error loading prompt: {e}")