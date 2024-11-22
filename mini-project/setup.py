import os

print('Making Virtual Environment')
os.system('python3 -m venv .venv && source .venv/bin/activate && pip install -qr requirements.txt')
print('All Done!')