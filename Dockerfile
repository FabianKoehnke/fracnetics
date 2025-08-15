FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY test/testI.py ..
WORKDIR /fracnetics

# Copy all files including pybind11 submodule
COPY . .

RUN pip install --upgrade pip
RUN pip install build scikit-build-core pybind11 twine

# Build with pybind11 from local submodule

# RUN cmake -Bbuild -S. && cmake --build build
RUN python3 -m build --sdist
RUN pip install dist/*.tar.gz
RUN python3 -u ../testI.py

CMD ["python3"]

