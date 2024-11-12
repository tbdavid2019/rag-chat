# Docker運行
docker-compose build .
docker-compose up -d
docker-compose logs -f



---- 
# 直接運行

pip install -U "huggingface_hub[cli]"
huggingface-cli login
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
pip install -r requirements.txt
python3 pdf_watcher.py
python3 app.py




