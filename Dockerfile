# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    build-essential python3 python3-dev python3-venv python3-pip \
    git wget libgl1 libglib2.0-0 libsm6 libxext6 \
    libjpeg-dev libpng-dev libxrender1 \
    ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev \
    libswscale-dev libavfilter-dev \
  && ln -sf /usr/bin/python3 /usr/bin/python \
  && ln -sf /usr/bin/pip3 /usr/bin/pip
RUN add-apt-repository ppa:savoury1/ffmpeg5
RUN add-apt-repository ppa:savoury1/ffmpeg6
RUN apt update
RUN apt install ffmpeg=7:6.*   
# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel --upgrade av
# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.3.43 --cuda-version 12.6 --nvidia

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client diffusers transformers
RUN uv pip install --no-cache-dir triton sageattention
RUN uv pip install --no-cache-dir git+https://github.com/openai/triton.git

# Add application code and scripts
ADD src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
# Set default model type if none is provided
ARG MODEL_TYPE="flux1-dev-fp8"

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip

# Download checkpoints/vae/unet/clip models to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -q -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -q -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -q -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "sd3" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
      wget -q -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors; \
    fi

RUN wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/diffusion_models/Wan2.1_T2V_14B_FusionX_VACE-Q8_0.gguf https://huggingface.co/QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF/resolve/main/Wan2.1_T2V_14B_FusionX_VACE-Q8_0.gguf?download=true; 
RUN wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/diffusion_models/ltx-video-2b-v0.9.safetensors https://huggingface.co/ali-vilab/VACE-LTX-Video-0.9/resolve/main/ltx-video-2b-v0.9.safetensors?download=true;
RUN wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/diffusion_models/Wan2.1_14B_VACE_F16.gguf https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/resolve/main/Wan2.1_14B_VACE-F16.gguf?download=true; 
RUN wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-Q8_0.gguf https://huggingface.co/QuantStack/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF/resolve/main/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-Q8_0.gguf?download=true; 
RUN wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/Wan2_1_VAE_bf16.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors?download=true;
RUN wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/text_encoders/umt5-xxl-encoder-Q8_0.gguf https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q8_0.gguf?download=true
# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Install custom nodes
RUN comfy-node-install comfyui-kjnodes comfyui-ic-light comfyui_ipadapter_plus comfyui_ipadapter_plus_v2 comfyui_essentials

RUN comfy-node-install comfy-image-saver comfyui-image-saver comfyui-impact-pack comfyui-impact-subpack ComfyUI_ADV_CLIP_emb comfyui-easy-use 

RUN comfy-node-install  comfyui-art-venture masquerade-nodes-comfyui comfyui-jakeupgrade


RUN comfy-node-install rgthree-comfy cg-use-everywhere comfyui_ultimatesdupscale ComfyUI-UltimateSDUpscale-GGUF ComfyUI-GGUF

RUN comfy-node-install comfyui_layerstyle teacachehunyuanvideo comfyui_controlnet_aux

RUN comfy-node-install ComfyUI-LTXVideo comfyui-videohelpersuite  comfyui-ltxvideolora

RUN comfy-node-install ComfyUI-WanVideoWrapper comfyui-rmbg comfyui-multigpu comfyui-custom-scripts

RUN git clone https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt.git custom_nodes/ComfyUI-Upscaler-Tensorrt
RUN pip install -r ./custom_nodes/ComfyUI-Upscaler-Tensorrt/requirements.txt
RUN git clone https://github.com/twri/sdxl_prompt_styler.git custom_nodes/sdxl_prompt_styler
RUN git clone https://github.com/sergekatzmann/ComfyUI_Nimbus-Pack.git custom_nodes/ComfyUI_Nimbus-Pack
RUN git clone https://github.com/Chaoses-Ib/ComfyUI_Ib_CustomNodes.git custom_nodes/ComfyUI_Ib_CustomNodes
RUN git clone https://github.com/ShmuelRonen/ComfyUI-VideoUpscale_WithModel.git custom_nodes/ComfyUI-VideoUpscale_WithModel
