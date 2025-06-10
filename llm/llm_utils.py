import openai

openai.api_base = "http://localhost:8001/v1"
openai.api_key = "EMPTY"  

def ask_local_llm(prompt: str, temperature: float = 0.2) -> str:
    response = openai.ChatCompletion.create(
        model="google/gemma-2b-it",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()
#response['choices'][0]['message']['content'].strip()
