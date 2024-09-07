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


#ifndef SPEC_FUNCTIONS_HPP_
#define SPEC_FUNCTIONS_HPP_

#include <iostream>
#include <vector>

#include "DlQuantization/Quantization.hpp"
#include "DlQuantization/EncodingRescale.hpp"

namespace DlQuantization
{

template <typename DTYPE>
void getRescaledOutputAndBiasImpl(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args, DTYPE* bias_out,
                           DTYPE* scaling_params, ComputationMode cpu_gpu_mode, bool withOffsetWrap);

template <typename DTYPE>
void getRescaledOutputAndBiasImplCpu(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &conv_args, DTYPE* bias_out,
                              DTYPE* scaling_params, bool withOffsetWrap);

// GPU implementations ...
#ifdef GPU_QUANTIZATION_ENABLED
template <typename DTYPE>
void getRescaledOutputAndBiasImplGpu(const DTYPE* bias_in, const int count, ConvSpecArgs<DTYPE> &hw_conv_args, DTYPE* bias_out,
                              DTYPE* scaling_params, bool withOffsetWrap);

#endif //End of GPU_QUANTIZATION_ENABLED

} // end of namespace DlQuantization

#endif // SPEC_FUNCTIONS_HPP_
