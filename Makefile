# 指定编译器
CC = gcc
NVCC = nvcc

# 通用编译选项
CFLAGS = -fPIC -I.
NVCCFLAGS = -shared -Xcompiler -fPIC

# 目标文件
OBJS = log.o list.o my_elf.o resource-mg.o util.o

# 输出库文件
TARGET1 = fake_libinit.so
TARGET2 = fake_libcudart.so

# 默认目标（同时生成两个动态库）
all: $(TARGET1) $(TARGET2)

# 生成 fake_libinit.so
$(TARGET1): cuda_init.c $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lelf -lcudart

# 生成 fake_libcudart.so
$(TARGET2): cuda_rthook.c $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lelf -lcudart

# 通用的 .o 文件规则
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(OBJS) $(TARGET1) $(TARGET2)

# 防止文件名与目标冲突
.PHONY: all clean
