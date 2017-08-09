CXXFLAGS= -Wall -Wextra -I boost_1_64_0 -march=native -std=gnu++0x -O3 -g -fopenmp -gstabs+ -g3
#CXXFLAGS+= -fsanitize=address
CXXFLAGS+= -I tbb-2017_U7/include -L tbb-2017_U7/build/linux_release/
LDLIBS=-ltbb
LDFLAGS=-Wl,-rpath=tbb-2017_U7/build/linux_release/


all: find_mds boost_1_64_0 tbb-2017_U7

boost_1_64_0:
	echo -e "\e[31;1mDownloading and unpacking Boost 1.64.\nThis will take a few minutes...\e[0m"
	wget -c https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2
	tar xf boost_1_64_0.tar.bz2

tbb-2017_U7:
	echo -e "\e[31;1mDownloading and compiling TBB 2017_U7.\nThis will take a few minutes...\e[0m"
	wget -c https://github.com/01org/tbb/archive/2017_U7.tar.gz
	tar xf 2017_U7.tar.gz
	make -C tbb-2017_U7 work_dir=build/linux MAKE="make RTM_KEY=''" # Old 'as' on cluster doesn't support rtm

clean:
	rm -fr tbb-2017_U7 boost_1_64_0 findmds

.PHONY: all clean
