#ifndef MATH
#define MATH

float factorial(uint n){
    float result = 1;
    for(uint i = 1; i <= n; i++){
        result *= float(i);
    }
    return result;
}

float C(uint n, uint m){
    return factorial(n) / factorial(n - m) / factorial(m);
}

float power(float m, uint n){
    float result = 1;
    for(uint i = 1; i <= n; i++){
        result *= m;
    }
    return result;
}

bool inOpenInterval(float left, float right, float x){
    return x < right && x > left;
}

#endif