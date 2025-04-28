####### Pre-Check #######
${FILE_PATH}
##### update pip libraries ######
python -m pip install --upgrade pip
pip list --outdated --format columns
pip install pip-review
pip-review --interactive

exit 1
