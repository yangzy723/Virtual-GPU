gcc -fPIC -c resource-mg.c   # 生成 resource-mg.o
gcc -fPIC -c log.c           # 生成 log.o
gcc -fPIC -c list.c          # 生成 list.o
gcc -fPIC -c util.c          # 生成 util.o
gcc -fPIC -c my_elf.c        # 生成 my_elf.o

nvcc -shared -Xcompiler -fPIC -o fake_libinit.so \
    cuda_init.c list.o my_elf.o util.o resource-mg.o log.o \
    -I. -lelf -lcudart

LD_PRELOAD=./fake_libinit.so python3 test/test.py

gcc -o resource-mg.o resource-mg.c log.c list.c -I.

nvcc -shared -Xcompiler -fPIC -o libmyelf.so my_elf.c util.c list.c -I. -lelf -lcudart

nvcc -shared -Xcompiler -fPIC -o fake_libinit.so cuda_init.c list.c util.c resource-mg.c -I. -lelf -lcudart