# StreamSplat (DSL-Lab, ICLR 2026, arXiv 2506.08862) @ 5d686e9, for the capacity study (LOG G4-G7).
# Build from the repository root, with the pinned clone as the context:
#   docker build -t streamsplat:5d686e9 -f docker/streamsplat/Dockerfile third_party/StreamSplat
# (reproduce.sh build does this). TORCH_CUDA_ARCH_LIST is the GPU's compute capability; 8.9 is the RTX 6000 Ada.
# Pins come from the clone's environment.yml (python 3.9.12, torch 2.4.1+cu124, 151 pip pins);
# all of them are kept (xformers too: the vendored DINOv2 layers use it when importable), and jaxtyping 0.2.36 is added
# (imported by utils/metrics.py, missing there).
# The image holds only the environment: code, weights and outputs are bind-mounted at run time.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_NO_CACHE=1
RUN uv venv --python 3.9.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
COPY environment.yml /opt/src/environment.yml
RUN sed -n '/- pip:/,$p' /opt/src/environment.yml | sed -n 's/^ *- //p' | grep -v -e '^pip:' \
      > /opt/src/requirements.txt \
 && uv pip install --python /opt/venv -r /opt/src/requirements.txt jaxtyping==0.2.36 \
      --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
COPY submodules/diff-gaussian-rasterization-orth /opt/src/diff-gaussian-rasterization-orth
ARG TORCH_CUDA_ARCH_LIST=8.9
RUN cd /opt/src/diff-gaussian-rasterization-orth \
 && TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST uv pip install --python /opt/venv --no-build-isolation . \
 && uv pip freeze --python /opt/venv > /opt/freeze.txt
