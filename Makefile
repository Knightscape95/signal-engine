# Makefile for C++ hardware implementation testing

CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3 -march=native
TARGET = test_manifold_engine
SRC_DIR = src/cpp
TEST_SRC = test_manifold.cpp

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $< -o $@

test: $(TARGET)
	./$(TARGET)

# ARM cross-compilation example
arm: $(TEST_SRC)
	arm-none-eabi-g++ -std=c++11 -mcpu=cortex-m4 -mthumb -O2 -I$(SRC_DIR) $< -o $(TARGET).elf

clean:
	rm -f $(TARGET) $(TARGET).elf *.o
