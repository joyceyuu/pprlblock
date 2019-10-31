FROM python:3.7

COPY . /

RUN pip install numpy pandas matplotlib tqdm memory_profiler

CMD ["python", "./comp_pb.py", "no"]
