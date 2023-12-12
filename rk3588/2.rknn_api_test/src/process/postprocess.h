
#ifndef RK3588_DEMO_POSTPROCESS_H
#define RK3588_DEMO_POSTPROCESS_H

#include <stdint.h>

#include <vector>

int get_top(float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount,
            uint32_t topNum);

#endif  // RK3588_DEMO_POSTPROCESS_H
