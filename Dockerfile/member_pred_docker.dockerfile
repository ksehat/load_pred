FROM python:3.7-slim-buster
WORKDIR C:\Users\Administrator\Desktop\Projects\member_pred
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "python", "api_member_pred.py" ]
CMD [ "python", "base_runner.py" ]