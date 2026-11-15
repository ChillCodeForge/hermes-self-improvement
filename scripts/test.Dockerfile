FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq git >/dev/null && rm -rf /var/lib/apt/lists/* \
 && git config --global user.email test@example.invalid && git config --global user.name test \
 && git config --global init.defaultBranch main && git config --global --add safe.directory '*'
RUN pip install -q uv
