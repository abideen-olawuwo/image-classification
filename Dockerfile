FROM python:3.8.12-slim

RUN pip install pipenv 

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"] 


# RUN pipenv install --system --deploy 



COPY ["predict.py", "model_C=1.0.bin" , "./"]



EXPOSE 3000


ENTRYPOINT ["waitress-serve" , "--listen=0.0.0.0:3000", "predict:app"]