CC := gcc

all: cudnn_perf.c
	gcc -o cudnn_perf cudnn_perf.c -lcudnn -lcudart

clean:
	rm cudnn_perf
