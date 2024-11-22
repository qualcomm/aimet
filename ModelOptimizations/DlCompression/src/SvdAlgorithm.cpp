//==============================================================================
//
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2017, Qualcomm Innovation Center, Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  3. Neither the name of the copyright holder nor the names of its contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
//
//==============================================================================
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <tuple>
#include "SvdAlgorithm.hpp"

namespace DlCompression
{
template <typename DTYPE>
ISVD<DTYPE>* GetSVDInstance()
{
    return new SVD_CORE<DTYPE>();
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::ComputeOriginalAndCompressedMemory_(SVD_COMPRESS_TYPE mode, int rows, int cols, int k_h, int k_w,
                                                          std::vector<unsigned int>& svd_ranks, size_t& original_size,
                                                          size_t& reduced_size)
{
    unsigned int r = 0, s = 0;
    original_size = (size_t) (rows * cols * k_h * k_w);
    if (!svd_ranks.size())
    {
        std::cerr << "Empty ranks set passed to method." << std::endl;
        throw std::runtime_error("Aborting");
    }

    r = svd_ranks.at(0);

    if (TYPE_SINGLE == mode)
    {
        reduced_size = size_t((rows + cols * k_h * k_w) * r);
    }

    else if ((TYPE_SUCCESSIVE == mode) && svd_ranks.size() > 1)
    {
        s            = svd_ranks.at(1);
        reduced_size = size_t((rows * r) + (r * s * k_h * k_w) + (s * cols));
    }
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::ComputeOriginalAndCompressedMAC_(SVD_COMPRESS_TYPE mode, int rows, int cols, int k_h, int k_w,
                                                       int act_h, int act_w, std::vector<unsigned int>& svd_ranks,
                                                       size_t& original_size, size_t& reduced_size)
{
    unsigned int r = 0, s = 0;
    size_t input_act_dim = act_h * act_w;
    original_size        = (size_t) (rows * cols * k_h * k_w * input_act_dim);

    if (!svd_ranks.size())
    {
        std::cerr << "Empty ranks set passed to method." << std::endl;
        throw std::runtime_error("Aborting");
    }

    r = svd_ranks.at(0);
    if (TYPE_SINGLE == mode)
    {
        reduced_size = size_t((rows + cols * k_h * k_w) * r * input_act_dim);
    }
    else if ((TYPE_SUCCESSIVE == mode) && svd_ranks.size() > 1)
    {
        s            = svd_ranks.at(1);
        reduced_size = size_t(((rows * r) + (r * s * k_h * k_w) + (s * cols)) * input_act_dim);
    }
}

template <typename DTYPE>
bool SVD_CORE<DTYPE>::ValidateRanksByMemory_(SVD_COMPRESS_TYPE mode, int rows, int cols, int k_h, int k_w,
                                             std::vector<unsigned int>& svd_ranks)
{
    size_t original_size, reduced_size;
    ComputeOriginalAndCompressedMemory_(mode, rows, cols, k_h, k_w, svd_ranks, original_size, reduced_size);
    return (reduced_size <= original_size);
}

template <typename DTYPE>
bool SVD_CORE<DTYPE>::ValidateRanksByMAC_(SVD_COMPRESS_TYPE mode, int rows, int cols, int k_h, int k_w, int act_h,
                                          int act_w, std::vector<unsigned int>& svd_ranks)
{
    size_t original_size, reduced_size;
    ComputeOriginalAndCompressedMAC_(mode, rows, cols, k_h, k_w, act_h, act_w, svd_ranks, original_size, reduced_size);
    return (reduced_size <= original_size);
}

template <typename DTYPE>
std::tuple<size_t, size_t>
SVD_CORE<DTYPE>::EstimateReducedCost_(typename std::map<std::string, LayerAttributes<DTYPE>>::iterator layer,
                                      int rank_index, bool useBestRanks)
{
    std::vector<int> shape = layer->second.shape;
    int M                  = shape.at(1);   // input channels
    int N                  = shape.at(0);   // output channels
    int k_h                = 1;             // Kernel height (default: 1 for FC layers)
    int k_w                = 1;             // Kernel width (default: 1 for FC layers)

    if (LAYER_TYPE_CONV == layer->second.layerType)
    {
        k_h = shape.at(2);
        k_w = shape.at(3);
    }
    std::vector<unsigned int> svdRanks;
    if (useBestRanks)
    {
        svdRanks = layer->second.bestRanks;
    }
    else
    {
        svdRanks = layer->second.candidateRanks.at(rank_index);
    }

    size_t original_size_mem, reduced_size_mem;
    size_t mem_savings;

    ComputeOriginalAndCompressedMemory_(layer->second.mode, M, N, k_h, k_w, svdRanks, original_size_mem,
                                        reduced_size_mem);

    DTYPE memCompressionScore = 1 - (DTYPE) reduced_size_mem / (DTYPE) original_size_mem;
    std::cout << "Compression ratio (memory) for: " << layer->first << " = " << memCompressionScore * 100 << " percent"
              << std::endl;
    mem_savings = original_size_mem - reduced_size_mem;

    size_t original_size_mac, reduced_size_mac;
    size_t mac_savings;
    int act_height = layer->second.activation_dims.first;
    int act_weight = layer->second.activation_dims.second;
    ComputeOriginalAndCompressedMAC_(layer->second.mode, M, N, k_h, k_w, act_height, act_weight, svdRanks,
                                     original_size_mac, reduced_size_mac);

    DTYPE macCompressionScore = 1 - (DTYPE) reduced_size_mac / (DTYPE) original_size_mac;
    std::cout << "Compression ratio (mac) for: " << layer->first << " = " << macCompressionScore * 100 << " percent"
              << std::endl;
    mac_savings = original_size_mac - reduced_size_mac;

    // store compression rate for each layer
    // compression rate using Memory and MAC should be same
    // condition makes sure that we don't store final compression rate twice
    if (!useBestRanks)
    {
        layer->second.compressionRate.push_back((memCompressionScore * 100));
    }
    return std::make_tuple(mem_savings, mac_savings);
}

bool sort_by_size(const std::pair<std::string, size_t>& A, const std::pair<std::string, size_t>& B)
{
    return A.second > B.second;
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::FillRankPool_(typename std::map<std::string, LayerAttributes<DTYPE>>::iterator layer,
                                    std::vector<std::vector<unsigned int>>& rankPool)
{
    std::vector<int> layerShape = layer->second.shape;
    unsigned int M              = layerShape.at(1);   // input channels
    unsigned int N              = layerShape.at(0);   // output channels
    unsigned int k_h            = 1;                  // Kernel height (defualt: 1 for FC layers)
    unsigned int k_w            = 1;                  // Kernel width (defualt: 1 for FC layers)
    unsigned int max_R = 0, max_S = 0;

    if (LAYER_TYPE_CONV == layer->second.layerType)
    {
        k_h = layerShape.at(2);
        k_w = layerShape.at(3);
    }
    max_R = std::min(M, N * k_h * k_w);

    if (TYPE_SINGLE == layer->second.mode)
    {
        // Further limit range of ranks based on
        // the criterion of compressibility.
        max_R = std::min(max_R, (M * N * k_h * k_w) / (M + N * k_h * k_w));
        for (unsigned int r = 1; r <= max_R; ++r)
        {
            std::vector<unsigned int> rank;
            rank.push_back(r);
            rankPool.push_back(rank);
        }
    }
    else if ((TYPE_SUCCESSIVE == layer->second.mode) && (LAYER_TYPE_CONV == layer->second.layerType))
    {
        for (unsigned int r = 1; r <= max_R; ++r)
        {
            max_S = std::min(N, r * k_h * k_w);
            for (unsigned int s = 1; s <= max_S; ++s)
            {
                std::vector<unsigned int> ranks;
                ranks.push_back(r);
                ranks.push_back(s);
                if (COST_TYPE_MEMORY == metric_)
                {
                    if (ValidateRanksByMemory_(layer->second.mode, M, N, k_h, k_w, ranks))
                    {
                        rankPool.push_back(ranks);
                    }
                }

                else if (COST_TYPE_MAC == metric_)
                {
                    int act_height = layer->second.activation_dims.first;
                    int act_weight = layer->second.activation_dims.second;
                    if (ValidateRanksByMAC_(layer->second.mode, M, N, k_h, k_w, act_height, act_weight, ranks))
                    {
                        rankPool.push_back(ranks);
                    }
                }
            }
        }
    }
}


std::tuple<cv::Mat, cv::Mat, cv::Mat> EigenSvd_(cv::Mat src)
{
    using Eigen::JacobiSVD;
    using Eigen::MatrixXf;
    using Eigen::ComputeThinU;
    using Eigen::ComputeThinV;

    // Convert cv::Mat to Eigen::Matrix
    MatrixXf src_(src.rows, src.cols);
    cv2eigen(src, src_);

    time_t startSvd, endSvd;
    time(&startSvd);
    JacobiSVD<MatrixXf> svd;
    svd.compute(src_, ComputeThinU | ComputeThinV);
    time(&endSvd);

    auto U_  = svd.matrixU();
    auto Vt_ = svd.matrixV().transpose().eval();
    auto W_  = svd.singularValues();

    cv::Mat U, W, Vt;
    U  = cv::Mat(U_.rows(),  U_.cols(),  CV_32F);
    Vt = cv::Mat(Vt_.rows(), Vt_.cols(), CV_32F);
    W  = cv::Mat(W_.rows(),  W_.cols(),  CV_32F);

    // Convert Eigen::Matrix to cv::Mat
    eigen2cv(U_, U);
    eigen2cv(Vt_, Vt);
    eigen2cv(W_, W);

    return std::make_tuple(U, W, Vt);
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::EstimateTAR_(typename std::map<std::string, LayerAttributes<DTYPE>>::iterator layer,
                                   std::vector<std::vector<unsigned int>>& rankPool,
                                   std::map<std::vector<unsigned int>, DTYPE>& TARMap)
{
#ifdef USE_OPENCV
    LayerAttributes<DTYPE> layerAttrib = layer->second;
    std::vector<int> shape             = layerAttrib.shape;
    int M                              = shape[1];   // number of inputchannels
    int N                              = shape[0];   // number of outputchannels
    int k_h                            = 1;
    int k_w                            = 1;
    std::cout << "Performing rank analysis on layer " << layer->first << std::endl;
    if (LAYER_TYPE_CONV == layerAttrib.layerType)
    {
        k_h = shape[2];
        k_w = shape[3];
    }
    int Nkk = N * k_h * k_w;

    // Compute SVD on src Matrix.
    cv::Mat srcMat(M, Nkk, CV_32F);
    TransposeSrcLayerWeights_(&layerAttrib, (DTYPE*) srcMat.datastart);
    cv::Mat U, W, VT;
    std::tie(U, W, VT) = EigenSvd_(srcMat);

    for (int i = 0; i < rankPool.size(); i++)
    {
        unsigned int r = 0, s = 0;
        auto ranks = rankPool.begin() + i;
        r          = ranks->at(0);
        if (TYPE_SUCCESSIVE == layerAttrib.mode)
            s = ranks->at(1);
        cv::Mat layerA_Mat(M, r, CV_32F);
        cv::Mat layerB_Mat(r, Nkk, CV_32F);
        // Truncate SVD components of src matrix based on rank 'r'
        TruncateMatrix_(U, W, VT, layerA_Mat, layerB_Mat, r);

        if (TYPE_SINGLE == layerAttrib.mode)
        {
            cv::Mat reconstructMat = layerA_Mat * layerB_Mat;
            // Estimate the tensor approximation residual (TAR)
            DTYPE recon_error = cv::norm(reconstructMat, srcMat, (cv::NORM_RELATIVE | cv::NORM_L2));
            TARMap.insert(std::make_pair(*ranks, recon_error));
            // TODO: Enable these logs in debug mode.
            // std::cout << "rank " << r << ": TAR = " << recon_error << std::endl;
        }
        else if ((TYPE_SUCCESSIVE == layerAttrib.mode) && (LAYER_TYPE_CONV == layerAttrib.layerType))
        {
            // First transpose layerB matrix from (r, NK^2) to (N, rk^2) form
            int rkk = r * k_h * k_w;   // Equivalent column dimension rk^2
            cv::Mat layerBT_Mat(N, rkk, CV_32F);
            cv::Mat layerB1_Mat(N, s, CV_32F);
            cv::Mat layerB2_Mat(s, rkk, CV_32F);

            Transpose_4DMatrix_((DTYPE*) (layerB_Mat.datastart), (DTYPE*) (layerBT_Mat.datastart), r, N, k_h, k_w);
            SVDCompress_(layerBT_Mat, layerB1_Mat, layerB2_Mat, s);

            cv::Mat layerB_Recon = layerB1_Mat * layerB2_Mat;
            // Now transpose back to (r, NK^2) form
            cv::Mat layerB_Recon_T(r, Nkk, CV_32F);
            Transpose_4DMatrix_((DTYPE*) (layerB_Recon.datastart), (DTYPE*) (layerB_Recon_T.datastart), N, r, k_h, k_w);

            // Finally reconstruct the original matrix from these parts.
            cv::Mat reconstructMat = layerA_Mat * layerB_Recon_T;
            DTYPE recon_error      = cv::norm(reconstructMat, srcMat, (cv::NORM_RELATIVE | cv::NORM_L2));
            TARMap.insert(std::make_pair(*ranks, recon_error));
            // TODO: Enable these logs in debug mode.
            // std::cout << "ranks (r, s) = (" << r << ", " << s << "): TAR = " << recon_error << std::endl;
        }
    }
#else
    // Silence compiler warnings
    (void) layer;
    (void) rankPool;
    (void) TARMap;
    throw std::runtime_error("This feature is available only with OPENCV support.");
#endif
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::PrintCandidateRanks(int rankIndex, bool useBestRanks)
{
    for (auto layer = LayerMap_.begin(); layer != LayerMap_.end(); ++layer)
    {
        std::vector<unsigned int> ranks;
        if (useBestRanks)
            ranks = layer->second.bestRanks;
        else if ((size_t) rankIndex < layer->second.candidateRanks.size())
            ranks = layer->second.candidateRanks.at(rankIndex);

        if (ranks.size())
        {
            unsigned int r = ranks.at(0);
            if (ranks.size() == 1)
                std::cout << layer->first << ": compressed with rank " << r << std::endl;

            else if (ranks.size() > 1)
            {
                unsigned int s = ranks.at(1);
                std::cout << layer->first << ": compressed with ranks = (" << r << ", " << s << ")" << std::endl;
            }
        }
    }
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::PickCandidateRanks_(std::map<std::vector<unsigned int>, DTYPE>& TARMap,
                                          std::vector<std::vector<unsigned int>>& candidate_ranks)
{
    // Store reference residuals
    int numRanks = candidate_ranks.size();
    std::vector<DTYPE> ref_residuals;
    for (int step = 1; step <= numRanks; ++step)
    {
        ref_residuals.push_back((DTYPE) step / numRanks);
    }
    // Create a vector to hold the smallest diff between
    // TARMap and reference residuals.
    // Initialize to highest possible normalized diff (100%)
    std::vector<DTYPE> delta(numRanks, 1.0);

    for (auto iter = TARMap.begin(); iter != TARMap.end(); ++iter)
    {
        for (int index = 0; index < numRanks; ++index)
        {
            if (fabs(ref_residuals.at(index) - iter->second) <= delta.at(index))
            {
                delta.at(index)           = fabs(ref_residuals.at(index) - iter->second);   // store lowest delta
                candidate_ranks.at(index) = iter->first;   // store ranks corresponding to this delta
            }
        }
    }
}

template <typename DTYPE>
int SVD_CORE<DTYPE>::SetCandidateRanks(int numCandidateRanks)
{
    if (numCandidateRanks < 1 || numCandidateRanks > MAX_CANDIDATE_RANKS)
    {
        numCandidateRanks = DEF_CANDIDATE_RANKS;
    }

    for (auto layer = LayerMap_.begin(); layer != LayerMap_.end(); ++layer)
    {
        std::vector<std::vector<unsigned int>> rankPool;
        std::map<std::vector<unsigned int>, DTYPE> TARMap;
        std::vector<std::vector<unsigned int>> candidate_ranks(numCandidateRanks);

        FillRankPool_(layer, rankPool);
        if (rankPool.size())
        {
            EstimateTAR_(layer, rankPool, TARMap);
            PickCandidateRanks_(TARMap, candidate_ranks);
            layer->second.candidateRanks = candidate_ranks;
        }

        else
        {
            numCandidateRanks = 0;   // No valid ranks available.
        }
    }
    return numCandidateRanks;
}

template <typename DTYPE>
std::vector<unsigned int>& SVD_CORE<DTYPE>::GetCandidateRanks(const std::string& layer_name, int rankIndex)
{
    auto it = LayerMap_.find(layer_name);
    if (it != LayerMap_.end())
    {
        if (rankIndex < 0 || rankIndex > MAX_CANDIDATE_RANKS)
        {
            std::cerr << "Invalid rank index " << rankIndex << std::endl;
            throw std::runtime_error("Aborting");
        }
        return it->second.candidateRanks.at(rankIndex);
    }
    else
    {
        std::cerr << "Layer with name " << layer_name << " not found in list." << std::endl;
        throw std::runtime_error("Aborting");
        // TODO: return invalid
    }
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::StoreBestRanks(const int rankIndex)
{
    for (auto layerIter = LayerMap_.begin(); layerIter != LayerMap_.end(); ++layerIter)
    {
        layerIter->second.bestRanks = layerIter->second.candidateRanks.at(rankIndex);
    }
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::StoreBestRanks(const std::string& layerName, const std::vector<unsigned int>& bestRanks)
{
    auto it = LayerMap_.find(layerName);
    if (it != LayerMap_.end())
    {
        it->second.bestRanks = bestRanks;
    }
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::ComputeNetworkCost()
{
    networkCost_Mem_ = 0;
    networkCost_Mac_ = 0;
    for (auto layerIter = LayerMap_.begin(); layerIter != LayerMap_.end(); ++layerIter)
    {
        if (layerIter->second.blobs.size())
        {
            size_t layerCost_Mem = layerIter->second.blobs[0].size();
            // Account for activations as well in MAC option
            size_t layerCost_Mac =
                layerCost_Mem * (layerIter->second.activation_dims.first * layerIter->second.activation_dims.second);
            networkCost_Mem_ += layerCost_Mem;
            networkCost_Mac_ += layerCost_Mac;
        }
    }
}

template <typename DTYPE>
DTYPE SVD_CORE<DTYPE>::GetCompressionScore(int rank_index, bool useBestRanks, size_t networkCostMem,
                                           size_t networkCostMac)
{
    size_t netSavings    = 0.0;
    size_t netMemSavings = 0.0;
    size_t netMacSavings = 0.0;
    for (auto layerIter = LayerMap_.begin(); layerIter != LayerMap_.end(); ++layerIter)
    {
        size_t mac_savings, mem_savings;
        std::tie(mem_savings, mac_savings) = EstimateReducedCost_(layerIter, rank_index, useBestRanks);
        netMemSavings += mem_savings;
        netMacSavings += mac_savings;
    }

    if ((networkCostMac == 0) || (networkCostMem == 0))
    {
        ComputeNetworkCost();
        networkCostMem = networkCost_Mem_;
        networkCostMac = networkCost_Mac_;
    }

    DTYPE memCompressionRatio = (DTYPE) netMemSavings / (DTYPE) networkCostMem;
    DTYPE macCompressionRatio = (DTYPE) netMacSavings / (DTYPE) networkCostMac;
    std::cout << "Compression ratio (mem) for network = " << memCompressionRatio * 100 << " percent" << std::endl;
    std::cout << "Compression ratio (mac) for network = " << macCompressionRatio * 100 << " percent" << std::endl;

    // Return net savings relative to total network cost
    if (metric_ == COST_TYPE_MAC)
    {
        return macCompressionRatio;
    }
    else
    {
        return memCompressionRatio;
    }
}

template <typename DTYPE>
LayerAttributes<DTYPE>* SVD_CORE<DTYPE>::GetLayerAttributes(const std::string& layer_name)
{
    auto it = LayerMap_.find(layer_name);
    if (it != LayerMap_.end())
        return &(it->second);
    else
        return NULL;
}

template <typename DTYPE>
SVD_COMPRESS_TYPE SVD_CORE<DTYPE>::GetCompressionType(const std::string& layer_name) const
{
    auto it = LayerMap_.find(layer_name);
    if (it != LayerMap_.end())
    {
        return it->second.mode;
    }
    else
        return TYPE_NONE;
}

template <typename DTYPE>
const std::vector<std::string> SVD_CORE<DTYPE>::GetLayerNames() const
{
    std::vector<std::string> layerNames;
    for (auto it: LayerMap_)
    {
        layerNames.push_back(it.first);
    }
    return layerNames;
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::Transpose_4DMatrix_(DTYPE* src, DTYPE* dst, int M, int N, int k_h, int k_w)
{
    int pos = 0, loc = 0;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            loc = j * N + i;
            for (int k = 0; k < k_h * k_w; ++k)
            {
                dst[pos++] = src[k_h * k_w * loc + k];
            }
        }
    }
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::TransposeSrcLayerWeights_(LayerAttributes<DTYPE>* layerAttrib, DTYPE* transposedBlob)
{
    if (layerAttrib)
    {
        std::vector<int> shape = layerAttrib->shape;
        int M                  = shape[1];   // number of inputchannels
        int N                  = shape[0];   // number of outputchannels
        int k_h                = 0;
        int k_w                = 0;

        if (LAYER_TYPE_CONV == layerAttrib->layerType)
        {
            // Fold 4D tensor along it's non-input modes
            // to transform the matrix shape from
            // (outputchannels, inputchannels, kernel_height, kernel_width) to
            // (inputchannels, outputchannels, kernel_height, kernel_width).
            k_h = shape[2];
            k_w = shape[3];
            Transpose_4DMatrix_((DTYPE*) (layerAttrib->blobs[0].data()), transposedBlob, N, M, k_h, k_w);
        }
        else if (LAYER_TYPE_FC == layerAttrib->layerType)
        {
#ifdef USE_OPENCV
            cv::Mat input_data(N, M, CV_32F, (DTYPE*) (layerAttrib->blobs[0].data()));
            cv::Mat output_data(M, N, CV_32F, transposedBlob);
            cv::transpose(input_data, output_data);
#endif
        }
    }
}

#ifdef USE_OPENCV

template <typename DTYPE>
void SVD_CORE<DTYPE>::SVDCompress_(cv::Mat& srcMat, cv::Mat& layerA_Mat, cv::Mat& layerB_Mat, unsigned int r)
{
    if (r > (unsigned int) std::min(srcMat.rows, srcMat.cols))
    {
        std::cerr << "Specified rank " << r << " is invalid." << std::endl;
        std::cerr << "srcMat.rows=" << srcMat.rows << ", srcMat.cols=" << srcMat.cols << std::endl;
        throw std::runtime_error("Aborting SVD compression");
    }

    // Decompose source matrix
    cv::Mat u, w, vt;
    std::tie(u, w, vt) = EigenSvd_(srcMat);

    u.colRange(0, r).copyTo(layerA_Mat);
    // Convert diagonal matrix w into normal matrix.
    cv::Mat w_matrix = cv::Mat::diag(w);
    // Compute w * vT after rank-based truncation
    // and store in layer B
    layerB_Mat = w_matrix(cv::Range(0, r), cv::Range(0, r)) * vt.rowRange(0, r);
}

#endif

#ifdef USE_OPENCV

template <typename DTYPE>
void SVD_CORE<DTYPE>::TruncateMatrix_(cv::Mat& U, cv::Mat& W, cv::Mat& VT, cv::Mat& layerA_Mat, cv::Mat& layerB_Mat,
                                      unsigned int r)
{
    if (r > (unsigned int) std::min(U.cols, VT.rows))
    {
        std::cerr << "Specified rank " << r << " is invalid." << std::endl;
        throw std::runtime_error("Aborting SVD compression");
    }

    // Slice 'r' colomns of U into layerA_Mat
    U.colRange(0, r).copyTo(layerA_Mat);

    // Convert diagonal matrix w into normal matrix.
    cv::Mat w_matrix = cv::Mat::diag(W);

    // Compute w * vT after rank-based truncation
    // with (rxr) sub-matrix of W and 'r' rows of VT into layerB_Mat.
    layerB_Mat = w_matrix(cv::Range(0, r), cv::Range(0, r)) * VT.rowRange(0, r);
}

#endif

template <typename DTYPE>
std::vector<std::vector<DTYPE>>&
SVD_CORE<DTYPE>::SplitLayerWeights(const std::string& layer_name, std::vector<std::vector<DTYPE>>& splitWeights,
                                   const std::vector<unsigned int>& weightSizes, const std::vector<unsigned int>& ranks)
{
    std::vector<DTYPE*> pSplitWeights;
    for (int i = 0; i < splitWeights.size(); ++i)
    {
        pSplitWeights.push_back(splitWeights[i].data());
    }
    SplitLayerWeights(layer_name, pSplitWeights, weightSizes, ranks);
    return splitWeights;
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::SplitLayerWeights(const std::string& layer_name, std::vector<DTYPE*> splitWeights,
                                        const std::vector<unsigned int>& weightSizes,
                                        const std::vector<unsigned int>& ranks)
{
#ifdef USE_OPENCV
    if (splitWeights.size() != ranks.size() + 1)
    {
        std::cerr << "splitWeights.size() = " << splitWeights.size() << ", ranks.size() = " << ranks.size()
                  << "; must have a rank for every pair of layer weights." << std::endl;
        throw std::runtime_error("Aborting SVD compression");
    }

    LayerAttributes<DTYPE>* layerAttrib = GetLayerAttributes(layer_name);
    if (!layerAttrib)
    {
        std::cerr << "No layer present in map with name " << layer_name << std::endl;
        throw std::runtime_error("Aborting SVD compression");
    }
    std::vector<int> shape = layerAttrib->shape;
    // Caffe stores blobs in row-major format as:
    // (outputchannels, inputchannels).
    // Create a duplicate source weight blob to
    // hold the transposed original in the form
    // (inputchannels, outputchannels).

    int M          = shape[1];   // number of inputchannels
    int N          = shape[0];   // number of outputchannels
    unsigned int r = ranks[0];   // input rank
    unsigned int s = 0;
    int k_h        = 1;
    int k_w        = 1;

    if ((TYPE_SUCCESSIVE == layerAttrib->mode) && ranks.size() > 1)
    {
        s = ranks[1];   // output rank
        if (!s)
        {
            std::cerr << ("No rank available for successive SVD.") << std::endl;
            throw std::runtime_error("Aborting SSVD compression");
        }
    }

    if (LAYER_TYPE_CONV == layerAttrib->layerType)
    {
        // Fold 4D tensor along it's non-input modes
        // to transform the matrix shape from
        // (outputchannels, inputchannels, kernel_height, kernel_width) to
        // (inputchannels, outputchannels, kernel_height, kernel_width).
        k_h = shape[2];
        k_w = shape[3];
    }
    int Nkk = N * k_h * k_w;   // the 2D column equivalent Nk^2

    // Create interim buffers for SVD computations
    cv::Mat srcMat(M, Nkk, CV_32F);
    cv::Mat layerA_Mat(M, r, CV_32F);
    cv::Mat layerB_Mat(r, Nkk, CV_32F);
    TransposeSrcLayerWeights_(layerAttrib, (DTYPE*) (srcMat.datastart));

    SVDCompress_(srcMat, layerA_Mat, layerB_Mat, r);

    // Create interim buffers for SSVD computations
    int rkk = r * k_h * k_w;   // Equivalent column dimension rk^2
    cv::Mat layerB1_Mat(N, s, CV_32F);
    cv::Mat layerB2_Mat(s, rkk, CV_32F);

    // Now store results into split layer blobs
    // in the same order as the original blob
    // {by transposition}

    if (LAYER_TYPE_CONV == layerAttrib->layerType)
    {
        // Layer A
        if (weightSizes[0] != (M * r))
        {
            std::cerr << "Allocated size for layer_A [" << weightSizes[0] << "] does not match computed size (" << M
                      << " x " << r << " x 1 x 1)" << std::endl;
            throw std::runtime_error("Aborting SVD compression");
        }
        Transpose_4DMatrix_((DTYPE*) (layerA_Mat.datastart), splitWeights[0], M, r, 1, 1);

        if (TYPE_SINGLE == layerAttrib->mode)
        {
            // Layer B
            if (weightSizes[1] != (r * Nkk))
            {
                std::cerr << "Allocated size for layer_B [" << weightSizes[1] << "] does not match computed size (" << r
                          << " x " << N << " x " << k_h << " x " << k_w << ")" << std::endl;
                throw std::runtime_error("Aborting SVD compression");
            }
            Transpose_4DMatrix_((DTYPE*) (layerB_Mat.datastart), splitWeights[1], r, N, k_h, k_w);
        }

        else if (TYPE_SUCCESSIVE == layerAttrib->mode)
        {
            // Perform successive SVD on Layer B

            // Create interim buffers for SVD computations
            cv::Mat layerBT(N, rkk, CV_32F);   // transpose of matrix B
            Transpose_4DMatrix_((DTYPE*) (layerB_Mat.datastart), (DTYPE*) (layerBT.datastart), r, N, k_h, k_w);

            // Split interim layer_B blob further
            // into layer_b1 & layer_b2 blobs.
            SVDCompress_(layerBT, layerB1_Mat, layerB2_Mat, s);

            // Copy weights back into layer B and layer C
            if (weightSizes[1] != (s * r * k_h * k_w))
            {
                std::cerr << "Allocated size for layer_B [" << weightSizes[1] << "] does not match computed size (" << s
                          << " x " << r << " x " << k_h << " x " << k_w << ")" << std::endl;
                throw std::runtime_error("Aborting SVD compression");
            }
            if (weightSizes[2] != (N * s))
            {
                std::cerr << "Allocated size for layer_C [" << weightSizes[2] << "] does not match computed size (" << N
                          << " x " << s << " x 1 x 1)" << std::endl;
                throw std::runtime_error("Aborting SVD compression");
            }
            memcpy(splitWeights[2], (DTYPE*) (layerB1_Mat.datastart),
                   weightSizes[2] * sizeof(DTYPE));   // "layer_b"
            memcpy(splitWeights[1], (DTYPE*) (layerB2_Mat.datastart),
                   weightSizes[1] * sizeof(DTYPE));   // "layer_c"
        }
    }
    else if (LAYER_TYPE_FC == layerAttrib->layerType)
    {
        // Layer A
        cv::Mat input_data_a(M, r, CV_32F, (DTYPE*) (layerA_Mat.datastart));
        cv::Mat output_data_a(r, M, CV_32F, (DTYPE*) (splitWeights[0]));
        cv::transpose(input_data_a, output_data_a);
        // Layer B
        cv::Mat input_data_b(r, N, CV_32F, (DTYPE*) (layerB_Mat.datastart));
        cv::Mat output_data_b(N, r, CV_32F, (DTYPE*) (splitWeights[1]));
        cv::transpose(input_data_b, output_data_b);
    }

    // In case this layer has a bias, we perform MSC: mean shift correction.
    if (2 == layerAttrib->blobs.size())
    {
        // Compute the low-rank approximation matrix. This is the product of the
        // decomposed matrices. Note that this matrix will be slightly different
        // than the original matrix 'srcMat'.
        cv::Mat lraMat(M, Nkk, CV_32F);
        if (TYPE_SINGLE == layerAttrib->mode)
        {
            lraMat = layerA_Mat * layerB_Mat;
        }

        else
        {
            // Re-combine matrices B1 and B2.
            cv::Mat layerB12_Mat(N, rkk, CV_32F);
            layerB12_Mat = layerB1_Mat * layerB2_Mat;
            // Transform the result.
            cv::Mat layerB12T(r, Nkk, CV_32F);
            Transpose_4DMatrix_((DTYPE*) (layerB12_Mat.datastart), (DTYPE*) (layerB12T.datastart), N, r, k_h, k_w);
            // Re-combine the result with the first matrix A.
            lraMat = layerA_Mat * layerB12T;
        }

        // Compute the error matrix: the difference between the original matrix and
        // the low-rank approximation.
        cv::Mat errorMat(M, Nkk, CV_32F);
        errorMat = srcMat - lraMat;

        // Perform mean shift correction (MSC) of the bias.
        // This method will store the bias correction in a class member. The bias
        // correction will then be used by SplitLayerBiases().
        MSC_(layer_name, ranks, errorMat);
    }

#else
    // Silence compiler warnings.
    (void) layer_name;
    (void) splitWeights;
    (void) weightSizes;
    (void) ranks;
    throw std::runtime_error("This feature is available only with OPENCV support.");
#endif
}

template <typename DTYPE>
std::vector<std::vector<DTYPE>>&
SVD_CORE<DTYPE>::SplitLayerBiases(const std::string& layer_name, std::vector<std::vector<DTYPE>>& splitBiases,
                                  const std::vector<unsigned int>& biasSizes, const std::vector<unsigned int>& ranks)
{
    std::vector<DTYPE*> pSplitBiases;
    for (int i = 0; i < splitBiases.size(); ++i)
    {
        pSplitBiases.push_back(splitBiases[i].data());
    }
    SplitLayerBiases(layer_name, pSplitBiases, biasSizes, ranks);
    return splitBiases;
}

template <typename DTYPE>
void SVD_CORE<DTYPE>::SplitLayerBiases(const std::string& layer_name, std::vector<DTYPE*> splitBiases,
                                       const std::vector<unsigned int>& biasSizes,
                                       const std::vector<unsigned int>& ranks)
{
    unsigned int r = 0, s = 0;

    LayerAttributes<DTYPE>* layerAttrib = GetLayerAttributes(layer_name);
    if (!layerAttrib)
    {
        std::cerr << "No layer present in map with name " << layer_name << std::endl;
        throw std::runtime_error("Aborting SVD compression");
    }

    if (layerAttrib->blobs.size() < 2)
    {
        // This layer has no bias terms. Returning with no processing.
        return;
    }

    if (splitBiases.size() != ranks.size() + 1)
    {
        std::cerr << "splitBiases.size() = " << splitBiases.size() << ", ranks.size() = " << ranks.size()
                  << "; must have a rank for every pair of layer biases." << std::endl;
        throw std::runtime_error("Aborting SVD compression");
    }

    r = ranks[0];
    if ((TYPE_SUCCESSIVE == layerAttrib->mode) && ranks.size() > 1)
    {
        s = ranks[1];   // output rank
        if (!s)
        {
            std::cerr << ("No rank available for successive SVD. Aborting!") << std::endl;
            throw std::runtime_error("Aborting SVD compression");
        }
    }
    // layer 'a': zero vector of size 'r'
    if (biasSizes[0] != r)
    {
        std::cerr << "Mismatch in bias vector dimensions! bias size " << biasSizes[0] << " should match rank " << r
                  << std::endl;
        throw std::runtime_error("Aborting SVD compression");
    }
    memset(splitBiases[0], 0, r * sizeof(DTYPE));

    if (TYPE_SINGLE == layerAttrib->mode)
    {
        // layer 'b': copy of original layer bias
        if (biasSizes[1] != layerAttrib->blobs[1].size())
        {
            std::cerr << "Mismatch in bias vector dimensions! bias size " << biasSizes[1]
                      << " should match original bias size " << layerAttrib->blobs[1].size() << std::endl;
            throw std::runtime_error("Aborting SVD compression");
        }
        memcpy(splitBiases[1], (DTYPE*) (layerAttrib->blobs[1].data()), layerAttrib->blobs[1].size() * sizeof(DTYPE));
    }
    else if (TYPE_SUCCESSIVE == layerAttrib->mode)
    {
        // layer 'b': zero vector of size 's'
        if (biasSizes[1] != s)
        {
            std::cerr << "Mismatch in bias vector dimensions! bias size " << biasSizes[1]
                      << " should match output rank " << s << std::endl;
            throw std::runtime_error("Aborting SVD compression");
        }
        memset(splitBiases[1], 0, s * sizeof(DTYPE));
        // layer 'c': copy of original layer bias
        if (biasSizes[2] != layerAttrib->blobs[1].size())
        {
            std::cerr << "Mismatch in bias vector dimensions! bias size " << biasSizes[2]
                      << " should match original bias size " << layerAttrib->blobs[1].size() << std::endl;
            throw std::runtime_error("Aborting SVD compression");
        }
        memcpy(splitBiases[2], (DTYPE*) (layerAttrib->blobs[1].data()), layerAttrib->blobs[1].size() * sizeof(DTYPE));
    }

    // Apply mean shift correction.
    // At this point, the bias from the original network was just copied into
    // the compressed network. To improve accuracy, we add the bias correction
    // to the original bias.
    std::vector<DTYPE> biasCorrection = GetBiasCorrection_(layer_name, ranks);
    // This index indicates which layer contains the bias. For SVD, the bias is
    // in the second layer. For SSVD, the bias is in the third layer.
    unsigned int indexBias = TYPE_SINGLE == layerAttrib->mode ? 1 : 2;
    // Add the bias correction.
    for (unsigned int i = 0; i < layerAttrib->blobs[1].size(); ++i)
    {
        splitBiases[indexBias][i] += biasCorrection[i];
    }
}

#ifdef USE_OPENCV

template <typename DTYPE>
void SVD_CORE<DTYPE>::MSC_(const std::string& layer_name, const std::vector<unsigned int>& ranks,
                           const cv::Mat& error_mat)
{
    LayerAttributes<DTYPE>* layerAttrib = GetLayerAttributes(layer_name);
    std::vector<int> shape              = layerAttrib->shape;
    int M                               = shape[1];   // number of inputchannels
    int N                               = shape[0];   // number of outputchannels
    int k_h                             = 1;          // kernel_height
    int k_w                             = 1;          // kernel_width
    if (LAYER_TYPE_CONV == layerAttrib->layerType)
    {
        k_h = shape[2];
        k_w = shape[3];
    }
    int Mkk = M * k_h * k_w;

    // Transpose the error matrix.
    cv::Mat errorT(N, Mkk, CV_32F);
    Transpose_4DMatrix_((DTYPE*) (error_mat.datastart), (DTYPE*) (errorT.datastart), M, N, k_h, k_w);

    // We want to support the case where the user didn't provide the input
    // channel mean. In that case, we just zero-init the input channel means,
    // which means the bias correction will get computed as an all-zero vector.
    if ((size_t) M != layerAttrib->inputChannelMean.size())
    {
        layerAttrib->inputChannelMean.clear();
        layerAttrib->inputChannelMean.resize(M, 0);
    }

    // Create a matrix with the mean of the input features.
    // For CONV layers, we need to replicate each value k^2 times.
    DTYPE* mean = layerAttrib->inputChannelMean.data();
    cv::Mat meanMat(Mkk, 1, CV_32F);
    // Go through all input feature maps.
    for (int m = 0; m < M; ++m)
    {
        // Replicate all values k^2 times.
        for (int k = 0; k < k_h * k_w; ++k)
        {
            meanMat.at<float>(m * k_h * k_w + k, 0) = mean[m];
        }
    }

    // Perform dot product between the transformed error matrix and the mean of
    // each input feature map.
    cv::Mat biasCorrectionMat(N, 1, CV_32F);
    biasCorrectionMat = errorT * meanMat;
    // Copy the result into a std::vector.
    std::vector<DTYPE> biasCorrection(N);
    memcpy(biasCorrection.data(), biasCorrectionMat.datastart, sizeof(float) * N);

    // Store the result.
    SetBiasCorrection_(layer_name, ranks, biasCorrection);
}

#endif

template <typename DTYPE>
void SVD_CORE<DTYPE>::SetBiasCorrection_(const std::string& layer_name, const std::vector<unsigned int>& ranks,
                                         const std::vector<DTYPE>& bias_correction)
{
    // Make sure the bias correction is of the same size as the original bias.
    size_t biasSize = bias_correction.size();
    if (GetLayerAttributes(layer_name)->blobs[1].size() != biasSize)
    {
        std::cerr << "Invalid bias size for layer " << layer_name << ": " << biasSize << std::endl;
        throw std::runtime_error("Aborting SVD compression");
    }
    // Copy the bias correction into 'BiasCorrection_'.
    BiasCorrection_[layer_name][ranks] = bias_correction;
}

template <typename DTYPE>
std::vector<DTYPE> SVD_CORE<DTYPE>::GetBiasCorrection_(const std::string& layer_name,
                                                       const std::vector<unsigned int>& ranks)
{
    // We want to support the use case where the user splits the bias before
    // the weights. If he does so, we don't have a bias correction.

    size_t expectedSize = GetLayerAttributes(layer_name)->blobs[1].size();
    // Check if we have a bias correction with the right size.
    if (BiasCorrection_.count(layer_name) && BiasCorrection_[layer_name].count(ranks) &&
        expectedSize == BiasCorrection_[layer_name][ranks].size())
    {
        // Return the bias correction.
        return BiasCorrection_[layer_name][ranks];
    }

    else
    {
        // So we don't have a bias correction at hand. Let's return an all-zero
        // vector instead. This means we won't do any correction.
        return std::vector<DTYPE>(expectedSize, 0);
    }
}

// Explicit Instantiations
template class SVD_CORE<float>;

template class SVD_CORE<double>;

template ISVD<float>* GetSVDInstance();

template ISVD<double>* GetSVDInstance();

}   // End of namespace DlCompression
