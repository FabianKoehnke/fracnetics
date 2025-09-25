CXX = clang++
CXXFLAGS = @compile_flags.txt
PYTHON_LIB = -L/Library/Frameworks/Python.framework/Versions/3.11/lib -lpython3.11

# Quellen
SRC_MAIN = src/main.cpp
SRC_IRIS = tests/iris.cpp

# Binaries
OUT_MAIN = src/main
OUT_IRIS = tests/iris

# Standard-Ziel
all: $(OUT_MAIN) $(OUT_IRIS)

# main Binary
$(OUT_MAIN): $(SRC_MAIN)
	$(CXX) $(CXXFLAGS) $(SRC_MAIN) $(PYTHON_LIB) -o $(OUT_MAIN)

# iris Binary
$(OUT_IRIS): $(SRC_IRIS)
	$(CXX) $(CXXFLAGS) $(SRC_IRIS) $(PYTHON_LIB) -o $(OUT_IRIS)

# Clean
clean:
	rm -f $(OUT_MAIN) $(OUT_IRIS)

