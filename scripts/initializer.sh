pip install -r requirements.txt

python -m spacy download en_core_web_md

wget https://github.com/qdrant/qdrant/releases/download/v1.12.5/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xvzf qdrant-x86_64-unknown-linux-gnu.tar.gz
rm qdrant-x86_64-unknown-linux-gnu.tar.gz