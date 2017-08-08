CXXFLAGS= -Wall -Wextra -I boost_1_64_0 -march=native -std=gnu++0x -O3 -g -fopenmp -gstabs+ -g3 #-fsanitize=address -O0
LDLIBS=-ltbb

all: find_mds boost_1_64_0

boost_1_64_0:
	echo -e "\e[31;1mDownloading and unpacking Boost 1.64.\nThis will take a few minutes...\e[0m"
	wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2
	tar xf boost_1_64_0.tar.bz2

.PHONY: all
