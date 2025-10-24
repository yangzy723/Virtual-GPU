#include <stdio.h>
#include <string.h>

int main(void) {
    const char *str1 = "apple";
    const char *str2 = "banana";
    const char *str3 = "apple";

    int result1 = strcmp(str1, str2);
    int result2 = strcmp(str1, str3);

    printf("strcmp(\"%s\", \"%s\") = %d\n", str1, str2, result1);
    printf("strcmp(\"%s\", \"%s\") = %d\n", str1, str3, result2);

    // 判断结果
    if (result1 < 0) {
        printf("\"%s\" < \"%s\"\n", str1, str2);
    } else if (result1 > 0) {
        printf("\"%s\" > \"%s\"\n", str1, str2);
    } else {
        printf("\"%s\" == \"%s\"\n", str1, str2);
    }

    if (result2 == 0) {
        printf("\"%s\" == \"%s\"\n", str1, str3);
    }

    return 0;
}
