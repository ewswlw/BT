# Bloomberg API Setup & Usage

## Install Bloomberg API
```bash
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi


# Git commands worth remembering 

git fetch origin
git reset --hard origin/master
git pull origin master

# Important commands
poetry run python data_pipelines/data_pipeline.py


# Notbooks to pdfs
poetry run jupyter nbconvert --to webpdf "your_notebook.ipynb"

# new command to run poetry now
poetry env activate