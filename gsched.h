#ifndef _GSCHED_H_
#define _GSCHED_H_

typedef struct _gsched_t {
    int (*init)(void);
    int (*retain)(int id);
    int (*release)(int id);
    int (*rm)(int id);
    void (*deinit)(void);
} gsched_t;

extern gsched_t *sched;     // 只声明
extern gsched_t sched_none; // 只声明

#define GSCHED_RETAIN sched->retain(1)
#define GSCHED_RELEASE sched->release(1)

#endif //_GSCHED_H_