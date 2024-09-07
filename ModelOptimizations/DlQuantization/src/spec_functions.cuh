//==============================================================================
//  @@-COPYRIGHT-START-@@
//
//  Copyright 2022 Qualcomm Technologies, Inc. All rights reserved.
//  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
//
//  The party receiving this software directly from QTI (the "Recipient")
//  may use this software as reasonably necessary solely for the purposes
//  set forth in the agreement between the Recipient and QTI (the
//  "Agreement"). The software may be used in source code form solely by
//  the Recipient's employees (if any) authorized by the Agreement. Unless
//  expressly authorized in the Agreement, the Recipient may not sublicense,
//  assign, transfer or otherwise provide the source code to any third
//  party. Qualcomm Technologies, Inc. retains all ownership rights in and
//  to the software
//
//  This notice supersedes any other QTI notices contained within the software
//  except copyright notices indicating different years of publication for
//  different portions of the software. This notice does not supersede the
//  application of any third party copyright notice to that third party's
//  code.
//
//  @@-COPYRIGHT-END-@@
//==============================================================================

#include <curand_kernel.h>
#include "DlQuantization/Quantization.hpp"

namespace DlQuantization
{
__device__ inline float clamp(float val, float min, float max)
{
    return fmaxf(fminf(val, max), min);
}

__device__ inline double clamp(double val, double min, double max)
{
    return fmax(fmin(val, max), min);
}

__device__ inline float round_nearest(float val)
{
    return roundf(val);
}

__device__ inline double round_nearest(double val)
{
    return round(val);
}

__device__ inline float round_downward(float val)
{
    return floorf(val);
}

__device__ inline double round_downward(double val)
{
    return floor(val);
}

__device__ inline float power(float x, float y)
{
    return powf(x, y);
}

__device__ inline double power(double x, double y)
{
    return pow(x, y);
}

__device__ float withoutOffsetWrapDevice(float offset, float requant_scale) {
    return 0;
}

__device__ float withOffsetWrapDevice(float offset, float requant_scale) {
    return offset / requant_scale;
}

typedef float(*offsetWrapPtr)(float, float);

__device__ offsetWrapPtr withoutOffsetHost = withoutOffsetWrapDevice;
__device__ offsetWrapPtr withOffsetHost = withOffsetWrapDevice;

template <typename DTYPE>
__device__ void getScaleAndBiasPerTensorDevice(const DTYPE* bias_in, const DTYPE acc_scale, const DTYPE out_offset,
                                               const DTYPE requant_scale, const int32_t bw, DTYPE* bias_out,
                                               offsetWrapPtr offset_func_ptr)
{
    DTYPE offsetWrapVal = offset_func_ptr(out_offset, requant_scale);
    DTYPE biasSim = round_nearest(*bias_in / acc_scale - offsetWrapVal);
    biasSim = round_downward(biasSim * power(2.0, 8.0 - bw));
    *bias_out = biasSim;
}

template <typename DTYPE>
__device__ void getScaleAndBiasPerChannelDevice(const DTYPE* bias_in, DTYPE acc_scale, DTYPE weight_scale,
                                                const DTYPE input_scale, DTYPE max_weight_scale, DTYPE out_scale,
                                                DTYPE* scaling_param, DTYPE out_offset, int32_t bw, DTYPE* bias_out,
                                                offsetWrapPtr offset_func_ptr)
{
    DTYPE accScaleCurr = weight_scale * input_scale;
    DTYPE normWeightScale = weight_scale / max_weight_scale;
    DTYPE requantScale = accScaleCurr / out_scale;
    *scaling_param = requantScale;
    DTYPE biasSim = round_nearest(*bias_in / accScaleCurr) * accScaleCurr;
    DTYPE offsetWrapVal = offset_func_ptr(out_offset, requantScale);
    biasSim = (biasSim / normWeightScale) / acc_scale - offsetWrapVal;
    biasSim = round_downward(biasSim * power(2.0, 8.0 - bw));
    *bias_out = biasSim;

}


}// End of namespace DlQuantization

