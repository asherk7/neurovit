import re
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def total_tokens(messages):
        return sum(count_tokens(msg['content']) for msg in messages)

def trim_chat_history(chat_history: list[dict], max_tokens: int = 4096) -> list[dict]:
    system_msgs = [msg for msg in chat_history if msg['role'] == 'system']
    user_assistant_msgs = [msg for msg in chat_history if msg['role'] != 'system']

    total = total_tokens(system_msgs + user_assistant_msgs)

    while total > max_tokens and user_assistant_msgs:
        del user_assistant_msgs[2:4]
        total = total_tokens(system_msgs + user_assistant_msgs)

    return system_msgs + user_assistant_msgs

def remove_markdown(text):
    text = re.sub(r"[*_`]", "", text)
    return text

def convert_history(history: list[dict]):
    lc_messages = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            role = 'human'
            lc_messages.append((role, content))

        elif role == "assistant":
            role = "ai"
            lc_messages.append((role, content))

        elif role == "system":
            lc_messages.append((role, content))

        else:
            raise ValueError(f"Unknown role: {role}")
        
    return lc_messages
