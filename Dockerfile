FROM python:3.11.3-slim-buster
WORKDIR C:\Project\member_pred\load_pred
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD [ "python", "api_member_pred.py" ]
CMD [ "python", "base_runner.py" ]