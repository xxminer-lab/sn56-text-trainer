# Text tournament trainer image.
# Base: the axolotl runtime image the default participant uses (torch 2.5.1 + cu124, transformers 4.x,
# flash-attn, peft, accelerate) - a known-good build target for the validator's docker build.
FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir huggingface_hub hf_transfer cryptography tenacity python-dotenv

RUN mkdir -p /workspace/scripts /app/checkpoints /cache
WORKDIR /workspace/scripts
COPY . /workspace/scripts
RUN chmod +x /workspace/scripts/entrypoint.sh
ENV PYTHONPATH=/workspace/scripts
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
