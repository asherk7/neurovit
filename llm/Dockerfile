FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /srv
RUN pip install vllm==0.3.3 --no-cache-dir

# if the model you want to serve requires you to accept the license terms,
# you must pass a HF_TOKEN environment variable, also ensure to pass a VLLM_API_KEY
# environment variable to authenticate your API
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server", \
            "--host", "0.0.0.0", "--port", "80", \
            "--model", "google/gemma-2b-it", \
            # depending on your GPU, you might or might not need to pass --dtype
            "--dtype=half"]