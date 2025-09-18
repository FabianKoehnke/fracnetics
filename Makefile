CXX = clang++
CXXFLAGS = @compile_flags.txt
PYTHON_LIB = -L/Library/Frameworks/Python.framework/Versions/3.11/lib -lpython3.11
SRC = src/main.cpp
OUT = src/main

all:
	$(CXX) $(CXXFLAGS) $(SRC) $(PYTHON_LIB) -o $(OUT)

clean:
	rm -f $(OUT)

