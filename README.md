created virtual environment:-
py -3.10 -m venv venv
venv\Scripts\activate


upgrading pip:-
python -m pip install --upgrade pip


pip install -r requirements.txt




to run api server:-
uvicorn app.main:app --reload


to test API:-
put the targeted language in line 17 of test_api.py file. Have a file named "test.mp3" in root directory.

python test_api.py



port to open:-
http://127.0.0.1.:8000/docs
