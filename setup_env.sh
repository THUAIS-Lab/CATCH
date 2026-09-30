set -x

# install uv
# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.local/bin/env

# export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu129

python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip install -e .[verl]

pip install e2b-code-interpreter python-dotenv
pip install lark-oapi openpyxl

# # install nodejs and e2b-cli for template delete
# curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
# apt install nodejs -y
# npm install -g @e2b/cli
