FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /srv
RUN pip install vllm==0.3.3 --no-cache-dir

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server", \
            "--host", "0.0.0.0", "--port", "80", \
            "--model", "google/gemma-2b-it", \
            "--dtype=half"]