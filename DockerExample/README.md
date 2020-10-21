## Dockerfile
```
FROM python:3.7

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY train_data.csv ./

CMD ["python", "./main.py", "/data/test_data.csv"]
```

## Build container
```
docker build -t linear_r .
```

## Run container
```
docker run -v /Users/rick/Desktop/ML_Docker/data/:/data/ linear_r
```

* "/Users/rick/Desktop/ML_Docker/data/:" is the path of the working dir while "/data/" is 
the folder containing test_data.csv
