ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  build-essential \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir boltz

FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV LANG=C.UTF-8 \
  PYTHONUNBUFFERED=1

ARG USERNAME=boltz
ARG UID=900
ARG GID=900

RUN groupadd --gid $GID $USERNAME && \
  useradd --uid $UID --gid $GID --create-home --shell /bin/bash $USERNAME

WORKDIR /app

COPY --chown=${USERNAME}:${USERNAME} LICENSE README.md /app/
COPY --chown=${USERNAME}:${USERNAME} examples /app/examples
COPY --chown=${USERNAME}:${USERNAME} scripts /app/scripts
COPY --chown=${USERNAME}:${USERNAME} docs /app/docs

USER $USERNAME

CMD ["boltz"]
