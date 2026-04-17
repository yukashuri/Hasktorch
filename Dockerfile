FROM htorch/hasktorch-jupyter:latest-cpu

# 1. パッケージインストールは root ユーザーで行う
USER root

# 1. 壊れている（404エラーになる）NodeSourceのリポジトリ設定を削除する
# 2. その後で update を実行する
RUN rm -f /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gnuplot \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Stack のインストール
RUN curl -sSL https://get.haskellstack.org/ | sh

# 3. 作業ユーザーを ubuntu に戻す
USER ubuntu
WORKDIR /home/ubuntu

# 4. ディレクトリ作成と移動
RUN mkdir -p libraries
# すでにイメージ内にあるファイルを移動（存在しないとエラーになるので注意）
RUN mv ./hasktorch ./inline-c ./dist-newstyle ./libraries/ || true

WORKDIR /home/ubuntu/libraries
RUN git clone https://github.com/DaisukeBekki/hasktorch-tools.git
RUN git clone https://github.com/DaisukeBekki/nlp-tools.git

WORKDIR /home/ubuntu/libraries/hasktorch-tools
# stack.yaml の書き換え
RUN sed -i -e "s|<path/to/your/hasktorch>|/home/ubuntu/libraries/hasktorch|g" stack.yaml

# 5. Cabal インストール（メモリ制限をかけつつ実行）
RUN cabal v1-install /home/ubuntu/libraries/hasktorch/hasktorch /home/ubuntu/libraries/hasktorch/codegen /home/ubuntu/libraries/hasktorch/libtorch-ffi /home/ubuntu/libraries/hasktorch/libtorch-ffi-helper --ghc-options "-j1 +RTS -A128m -n2m -RTS"

WORKDIR /home/ubuntu
RUN cabal v1-install ./libraries/hasktorch-tools ./libraries/nlp-tools --ghc-options "-j1 +RTS -A128m -n2m -RTS"