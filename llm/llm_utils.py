import re
import tiktoken

# Load GPT-2 tokenizer for token counting
tokenizer = tiktoken.get_encoding("gpt2")

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a given text using GPT-2 tokenizer.

    Args:
        text (str): The input text.

    Returns:
        int: Number of tokens in the text.
    """
    return len(tokenizer.encode(text))

def total_tokens(messages: list[dict]) -> int:
    """
    Calculate the total token count for a list of chat messages.

    Args:
        messages (list[dict]): List of messages (each with a 'content' field).

    Returns:
        int: Total number of tokens across all messages.
    """
    return sum(count_tokens(msg['content']) for msg in messages)

def trim_chat_history(chat_history: list[dict], max_tokens: int = 4096) -> list[dict]:
    """
    Trim chat history to fit within the max token limit.

    Keeps all 'system' messages and trims 'user' and 'assistant' messages by removing
    the 3rd and 4th elements (typically the oldest messages) until under the limit.

    Args:
        chat_history (list[dict]): The full chat history.
        max_tokens (int): The maximum number of tokens allowed.

    Returns:
        list[dict]: Trimmed chat history within the token limit.
    """
    system_msgs = [msg for msg in chat_history if msg['role'] == 'system']
    user_assistant_msgs = [msg for msg in chat_history if msg['role'] != 'system']

    total = total_tokens(system_msgs + user_assistant_msgs)

    while total > max_tokens and user_assistant_msgs:
        del user_assistant_msgs[2:4]  # Remove 3rd and 4th messages to maintain system messages
        total = total_tokens(system_msgs + user_assistant_msgs)

    return system_msgs + user_assistant_msgs

def remove_markdown(text: str) -> str:
    """
    Remove common markdown formatting from text.

    Args:
        text (str): Input string possibly containing markdown.

    Returns:
        str: Cleaned text with markdown removed.
    """
    return re.sub(r"[*_`]", "", text)

def convert_history(history: list[dict]) -> list[tuple[str, str]]:
    """
    Convert chat history format for LangChain chat models.

    Args:
        history (list[dict]): List of dicts with 'role' and 'content'.

    Returns:
        list[tuple[str, str]]: Reformatted messages as (role, content) tuples
                               where role is 'human', 'ai', or 'system'.
    """
    lc_messages = []

    for msg in history:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            lc_messages.append(("human", content))
        elif role == "assistant":
            lc_messages.append(("ai", content))
        elif role == "system":
            lc_messages.append(("system", content))
        else:
            raise ValueError(f"Unknown role: {role}")

    return lc_messages
