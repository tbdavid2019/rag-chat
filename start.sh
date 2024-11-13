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



---
nohup /home/ec2-user/myenv/bin/python3 pdf_watcher.py > pdf_watcher.log 2>&1 &
nohup /home/ec2-user/myenv/bin/python3 app.py > app.log 2>&1 &
ps aux | grep python3
