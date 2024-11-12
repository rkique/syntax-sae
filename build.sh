pip install -U pip setuptools wheel
pip install -U spacy
echo SPACY INSTALLED
python -m spacy download en_core_web_sm
pip install -r requirements.txt
