all: dish_bench relu_bench

%: %.c
	gcc -o $@ -O3 -ffast-math -mavx2 -Wall $<

clean:
	rm -f relu_bench dish_bench
