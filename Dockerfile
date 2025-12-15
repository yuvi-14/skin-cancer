FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

RUN chmod -R 777 /code

CMD ["gunicorn", "-b", "0.0.0.0:7860", "predict_flask:app"]