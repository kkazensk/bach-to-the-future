####### Pre-Check #######
${FILE_PATH}
##### update pip libraries ######
python3 -m venv bachvenv
source bachvenv/bin/activate
python -m pip install --upgrade pip 
python3 -m pip install -r requirements.txt

exit 1
