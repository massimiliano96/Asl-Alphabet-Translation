FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0

RUN python -m pip install --upgrade pip

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 9090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9090"]


