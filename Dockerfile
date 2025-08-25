FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /_fracnetics

# Copy all files including pybind11 submodule
COPY . .

RUN pip install --upgrade pip
RUN pip install build scikit-build-core pybind11 twine

# RUN cmake -Bbuild -S. && cmake --build build
RUN python3 -m build 
RUN pip install dist/*.whl
#RUN pip install .

#CMD ["twine", "upload", "dist/*"]
CMD ["python3"]

