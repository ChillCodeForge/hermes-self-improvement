FROM python:3.12-slim
# git: candidate/promotion repositories; acl: setfacl/getfacl for provision.sh;
# util-linux/passwd provide setpriv, runuser, useradd. sudo is deliberately absent.
RUN apt-get update -qq && apt-get install -y -qq git acl util-linux passwd >/dev/null \
 && rm -rf /var/lib/apt/lists/* \
 && if command -v sudo >/dev/null; then echo "sudo must not be installed in the test image" >&2; exit 1; fi \
 && git config --global user.email test@example.invalid && git config --global user.name test \
 && git config --global init.defaultBranch main && git config --global --add safe.directory '*'
RUN pip install -q uv
