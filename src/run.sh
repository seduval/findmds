#! /bin/sh
module add tbb
rm find_mds
make
# Limit available memory
ulimit -v `vmstat | tail -1 | awk '{ print int(0.99*($4+$5+$6)) }'`
./find_mds
