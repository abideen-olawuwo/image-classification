docker run -it python 3.8.12-slim

docker run -it --entrypoint=bash python:3.8.12-slim

apt-get update

apt-get install wget

mkdir test

cd test

docker build -t converted-test .

docker run -it --rm --entrypoint=bash converted-test

pipenv install

Dockerfile (RUN pipenv install --system --deploy)

docker build -t converted-test .

"""""""""""""""""""'""

docker run -it --rm --entrypoint=bash converted-test

ls

gunicorn --bind=0.0.0.0:3000 converted:app

pipenv install waitress

pipenv shell

waitress-serve --listen=0.0.0.0:3000 predict:app

docker run -it --rm -p 3000:3000 converted-test

pipenv shell
  
waitress-serve --listen=0.0.0.0:3000 predict:app
