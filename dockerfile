FROM python:3.11.0rc2-slim-bullseye

WORKDIR /home/_code_/other_stuff/customer_review_classifier/MLFlow_learning

COPY entrypoint.sh .
COPY requirements.txt .

RUN apt update && apt upgrade -y

RUN apt -y install postgresql postgresql-contrib nginx

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 8080

ENTRYPOINT ["/home/_code_/other_stuff/customer_review_classifier/MLFlow_learning/entrypoint.sh"]