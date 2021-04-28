all: 01-nbody-ori.cu 02-nbody-opt1.cu 03-nbody-opt2.cu
	nvcc -std=c++11 -o 01-nbody-ori 01-nbody-ori.cu
	nvcc -std=c++11 -o 02-nbody-opt1 02-nbody-opt1.cu
	nvcc -std=c++11 -o 03-nbody-opt2 03-nbody-opt2.cu
	nvcc -std=c++11 -o 04-nbody-shuffle 04-nbody-shuffle.cu
clean:
	rm 01-nbody-ori 02-nbody-opt1 03-nbody-opt2 04-nbody-shuffle
test:
	@echo 02-nbody-opt1:
	@./02-nbody-opt1 11
	@./02-nbody-opt1 15
	@echo
	@echo 03-nbody-opt2:
	@./03-nbody-opt2 11
	@./03-nbody-opt2 15
	@echo
	@echo 04-nbody-shuffle:
	@./04-nbody-shuffle 11
	@./04-nbody-shuffle 15
