matrices = $(wildcard matrices_paar_header/*.h)
targets = $(matrices:.h=_check)
runs = $(matrices:.h=_run)
slpprograms = slp_heuristic

all: slp_heuristic $(targets)
run: $(runs)

$(targets): %_check: paar.cpp
	g++ $< -o $@_alg1 -O3 -g -Wall -std=c++11 -mpopcnt -D PAAR1 -include stdint.h -include $*.h
	g++ $< -o $@_alg2 -O3 -g -Wall -std=c++11 -mpopcnt -D PAAR2 -include stdint.h -include $*.h

$(runs): %_run:
	timeout 3h $*_check_alg1 > $*_run_alg1
	timeout 3h $*_check_alg2 > $*_run_alg2

$(slpprograms): %: %.cpp
	g++ -o $@ $< -std=c++11 -O3 -Wall

clean:
	${RM} paar_header/*_check_alg1 paar_header/*_check_alg2
	${RM} $(slpprograms)

.PHONY: clean
