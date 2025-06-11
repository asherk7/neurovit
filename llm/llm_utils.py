from openai import OpenAI
import re

client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
chat_history = [
    {"role": "user", "content": "What is your purpose?"},
    {"role": "assistant", "content": "I am a helpful assistant for medical brain tumor information."},
    {"role": "user", "content": "Respond clearly and concisely using plain text only. Do not use markdown, bold, italic, or any special formatting."},
    {"role": "assistant", "content": "Of course, I will respond to the best of my abilities."},
    {"role": "user", "content": "Answer medical questions to the best of your ability, then mention at the end of your response to seek professional advice for further clarification."},
    {"role": "assistant", "content": "Of course, I will respond to the best of my abilities."},
    {"role": "user", "content": "I've received my MRI brain scan results, and I have a Glioma Tumor, could you elaborate on what it is?"},
    {"role": "assistant", "content": "A Glioma is a type of brain tumor that starts in the brain or spinal cord. Symptoms of a Glioma may include headaches, seizures, vision problems, speech problems, and changes in mood or behavior. If you have any concerns about a brain tumor, please consult a medical professional."},
]

def ask_local_llm(user_msg: str) -> str:
    chat_history.append({"role": "user", "content": user_msg})
    response = client.chat.completions.create(
        model="google/gemma-2b-it",
        messages=chat_history,
        temperature=0.2,
        max_tokens=128,
    )
    reply = response.choices[0].message.content.strip()
    clean_response = remove_markdown(reply)
    chat_history.append({"role": "assistant", "content": reply})
    return clean_response

def remove_markdown(text):
    text = re.sub(r"[*_`]", "", text)
    return text
