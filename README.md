LD_PRELOAD=./libmyelf.so python3 test.py

nvcc -shared -Xcompiler -fPIC -o libmyelf.so my_elf.c util.c list.c -I. -lelf -lcudart