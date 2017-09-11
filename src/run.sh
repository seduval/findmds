#! /bin/sh
module add gcc/5.3.0
rm find_mds
make
./find_mds
