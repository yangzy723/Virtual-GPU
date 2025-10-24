# 指定编译器
CC = gcc
NVCC = nvcc

# 通用编译选项
CFLAGS = -fPIC -I.
NVCCFLAGS = -shared -Xcompiler -fPIC

# 目标文件
OBJS = resource-mg.o log.o list.o util.o my_elf.o
TARGET = fake_libinit.so

# 默认目标
all: $(TARGET)

# 生成共享库
$(TARGET): cuda_init.c $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lelf -lcudart

# 生成 .o 文件的通用规则
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(OBJS) $(TARGET)

# 防止文件名与目标冲突
.PHONY: all clean
