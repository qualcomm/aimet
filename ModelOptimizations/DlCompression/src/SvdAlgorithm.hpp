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

#ifndef SVDALGORITHM_H
#define SVDALGORITHM_H

#include "DlCompression/ISVD.hpp"
#include <map>
#include <string>
#include <vector>
#include <Eigen/SVD>

#ifdef USE_OPENCV

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>

#endif   // USE_OPENCV

namespace DlCompression
{
// A 'good' number of candidate ranks that represent
// the TAR range (0, 1) with sufficient granularity.
#define DEF_CANDIDATE_RANKS 20

// A limit on the maximum number of candidate rank sets
// to evaluate network performance within a reasonable
// amount of time.
#define MAX_CANDIDATE_RANKS 50

// The threshold for the top N layers to be picked for
// compression that account to a certain percentage of
// the total network cost.
#define LAYER_SELECTION_THRESHOLD 0.60

// A heuristic threshold for the minimum matrix dimensions
// for SVD compression.
#define MIN_LAYER_DIM_FOR_SVD 10


template <typename DTYPE>
class SVD_CORE : public ISVD<DTYPE>
{
public:
    /**
     * @brief Set the preferred list of ranks for compression analysis.
     * @param numCandidateRanks Number of potential ranks with which
     *  to try layer compression across the network.
     *  Evaluate all possible combination of rank values for compressing
     *  all applicable layers in the network, and store potential ranks
     *  to be used for analysimg network performance after compression.
     *  \return Actual number of candidate ranks specified by SVD.
     */
    virtual int SetCandidateRanks(int numCandidateRanks) override;

    /**
     * @brief Get candidate ranks for a given layer and a specified rank index
     * @param layer_name Layer name
     * @param rankIndex The particular set of ranks for the layer.
     */
    virtual std::vector<unsigned int>& GetCandidateRanks(const std::string& layer_name, int rankIndex) override;

    /**
     * @brief Print candidate ranks of all layers for a given rank index.
     * @param rankIndex Common Index specifying a set of ranks across all layers.
     * @param useBestRanks Print only best ranks of a layer
     *  The function displays the candidate ranks for compression across
     *  all layers of the network which is usefu diagnostic information
     *  when evaluating network performance with multiple candidates.
     *  The function also displays the best ranks selected for each layer
     *  which is useful information during the final compression stage.
     */
    virtual void PrintCandidateRanks(int rankIndex, bool useBestRanks) override;

    /**
     * @brief Determine layer type for SVD Compression (CONV/FC).
     * @param layer_type String form of the layer type.
     *  The function returns the enum equivalent of the string
     *  denoting the layer type.
     */
    virtual COMPRESS_LAYER_TYPE GetLayerType(const std::string& layer_type) override
    {
        if (("Convolution" == layer_type) || ("Convolutional" == layer_type))
            return LAYER_TYPE_CONV;
        else if (("InnerProduct" == layer_type) || ("FullyConnected" == layer_type))
            return LAYER_TYPE_FC;
        else
            return LAYER_TYPE_OTHER;
    }

    /**
     * @brief Return a list of all layer names.
     */
    virtual const std::vector<std::string> GetLayerNames() const override;

    /**
     * @brief Determine Compression type (SVD/SSVD).
     * @param layer_type Layer type (CONV/FC).
     * @param svd_pass_type String form of compression type
     *  The function returns the enum equivalent of the string
     *  denoting the desired compression type (SVD/SSVD) based on
     *  the layer type.
     */
    SVD_COMPRESS_TYPE GetCompressionType(COMPRESS_LAYER_TYPE layer_type, const std::string& svd_pass_type) const
    {
        if ("single" == svd_pass_type)
        {
            return TYPE_SINGLE;
        }
        else if ("successive" == svd_pass_type)
        {
            if (LAYER_TYPE_FC == layer_type)
            {
                std::cout << "SSVD not supported on FC layers. Switching to SVD." << std::endl;
                return TYPE_SINGLE;   // SSVD not applicable for FC layers
            }
            else
            {
                return TYPE_SUCCESSIVE;
            }
        }
        else
        {
            return TYPE_NONE;
        }
    }

    /**
     * @brief Determine Compression type (SVD/SSVD) of stored layer.
     * @param layer_name Name of stored layer.
     *  This overloaded function returns the compression type of a
     *  layer already stored in the layer map.
     */
    virtual SVD_COMPRESS_TYPE GetCompressionType(const std::string& layer_name) const override;

    /**
     * @brief Store the cost metric for evaluating degree of compression.
     * @param metric The cost type: can be MEMORY/MACs.
     */
    virtual void SetCostMetric(NETWORK_COST_METRIC metric)
    {
        metric_ = metric;
    }

    /**
     * @brief Store layer attributes.
     * @param layerName 'key' used to store in map
     * @param layerAttrib 'value' stored against the key in map
     */
    virtual void StoreLayerAttributes(const std::string& layerName, const LayerAttributes<DTYPE>& layerAttrib) override
    {
        LayerMap_.insert(make_pair(layerName, layerAttrib));
    }

    /**
     * @brief Compute the total cost of the network
     *  based on the cost metric.
     *  computed accounting for contributions from all compressible layers.
     */
    virtual void ComputeNetworkCost() override;

    /**
     * @brief Retrieve layer attributes.
     * @param layer_name Name of layer to look in map
     * @return LayerMap for the corresponding layer name.
     */
    virtual LayerAttributes<DTYPE>* GetLayerAttributes(const std::string& layer_name) override;

    /**
     * @brief Get relative compression score with a given rank index
     * @param rank_index Index of the vector of candidate ranks.
     * @param useBestRanks Use LayerAttributes --> bestRanks to
     *  compute compression score (used during direct compression)
     *  instead of ranks indexed by rank_index (used in analysis phase).
     *  The method the relative compression for each layer of the network
     *  as a measure of the fraction of removed values that is achieved
     *  by the specified set of ranks.
     */
    virtual DTYPE GetCompressionScore(int rank_index, bool useBestRanks, size_t networkCostMem,
                                      size_t networkCostMac) override;

    /**
     * @brief Split layer weight matrix into residual sub-matrices using SVD/SSVD.
     * @param layer_name Name of layer to split.
     * @param splitWeights The vector of resultant weights of split layers
     * @param weightSizes Sizes of weight vectors
     * @param rank(s) to be used for compression.
     * The method splits a matrix M[mxn] using SVD and truncates the
     * resulting [U, SVt] matrices using rank r of sizes [mxr], [rxn] respectively.
     * The method is applied successively on the [SVt] matrix in the case of
     * successive SVD.
     * ALGORITHM:
     ********************************************************************
     * SVD on Inner Product Layer
     * Split a [mxn] matrix into two sub-matrices of size
     * [mxr] and [rxn]
     * where
     * m = number of input channels
     * n = number of output channels
     * r = compression rank
     ********************************************************************
     * SVD on Convolution layer
     * Split a [mxnxkxk] matrix into two sub-matrices of size
     * [mxrx1x1] and [rxnxkxk]
     * where
     * m = number of input channels
     * n = number of output channels
     * k = kernel size (for square kernels, but applicable to
     *                  non-square kernels as well)
     * r = compression rank
     ********************************************************************
     * SSVD on Convolution layer (not applicable to other layer types)
     * Split a [mxnxkxk] matrix into three (or more) sub-matrices of size
     * [mxrx1x1], [rxsxkxk] and [sxnx1x1]
     * where
     * m = number of input channels
     * n = number of output channels
     * k = kernel size (for square kernels, but applicable to
     *                  non-square kernels as well)
     * r,s = <input, output> compression ranks
     ********************************************************************
     *
     * As part of splitting the weights, this method computes the bias
     * correction according to the MSC algorithm.
     */
    virtual std::vector<std::vector<DTYPE>>& SplitLayerWeights(const std::string& layer_name,
                                                               std::vector<std::vector<DTYPE>>& splitWeights,
                                                               const std::vector<unsigned int>& weightSizes,
                                                               const std::vector<unsigned int>& ranks) override;

    virtual void SplitLayerWeights(const std::string& layer_name, std::vector<DTYPE*> splitWeights,
                                   const std::vector<unsigned int>& weightSizes,
                                   const std::vector<unsigned int>& ranks) override;

    /**
     * @brief Split layer bias vector into residual sub-vectors using SVD/SSVD.
     * @param layer_name Name of layer to split.
     * @param splitBiases The vector of resultant biases of split layers
     * @param biasSizes Sizes of bias vectors
     * @param rank(s) to be used for compression.
     * The method operates on biases of the SVD-split layers by
     * truncating the bias vector of the first layer to size 'r' (rank) and
     * zeroing out its contents and simply copies the original bias vector
     * for the second layer. It applies this procedure successively for
     * subsequent layers for SSVD compression.
     * ALGORITHM:
     ********************************************************************
     *                               SVD
     * Split a bias vector b into two sub-vectors of the form:
     * b1: A zero-vector of size of the input rank
     * b2: The copy of the original bias vector
     ********************************************************************
     *                              SSVD
     * Split a bias vector b into three sub-vectors of the form:
     * b1: A zero-vector of size of the input rank (r)
     * b2: A zero-vector of size of the output rank (s)
     * b3: The copy of the original bias vector
     ********************************************************************
     *
     * Prerequisite: Call SplitLayerWeights() first. The reason we need to call
     * SplitLayerWeights() first is because it computes the bias correction.
     */
    virtual std::vector<std::vector<DTYPE>>& SplitLayerBiases(const std::string& layer_name,
                                                              std::vector<std::vector<DTYPE>>& splitBiases,
                                                              const std::vector<unsigned int>& biasSizes,
                                                              const std::vector<unsigned int>& ranks) override;

    virtual void SplitLayerBiases(const std::string& layer_name, std::vector<DTYPE*> splitBiases,
                                  const std::vector<unsigned int>& biasSizes,
                                  const std::vector<unsigned int>& ranks) override;

    /**
     * @brief Store rank(s) of all layers corresponding to an index from candidateRanksMap_.
     * @param rankIndex Common Index specifying a set of ranks across all layers
     * contributing to a finite approximation residual.
     * The method stores the best ranks for compression across all layers in LayerMap_
     * based on the index from the candidateRanksMap_.
     * This method works during network performance analysis with
     * multiple candidate ranks.
     */
    virtual void StoreBestRanks(const int rankIndex) override;

    /**
     * @brief Store rank(s) of a single layer corresponding to layer_name in map.
     * @param layerName Name of layer to look in map.
     * @param bestRanks Set of ranks that are found to compress the layer best.
     * This method works best in the non-evaluation mode where best ranks
     * for a layer are pre-determined.
     */
    virtual void StoreBestRanks(const std::string& layerName, const std::vector<unsigned int>& bestRanks) override;

private:
    /**
     * @brief Compute the original and reduced sizes of a layer with a given set of ranks.
     * @param mode Compression mode (SVD/SSVD)
     * @param rows Number of rows of equivalent 2D matrix.
     * @param cols Number of columns of equivalent 2D matrix.
     * @param k_h Kernel height (for CONV layer)
     * @param k_w Kernel width (for CONV layer)
     * @param svd_ranks The vector of ranks with which layer is compressed.
     * @param original_size Original layer size
     * @param reduced_size Reduced_size layer size
     *  The method computes the original size of the weights of a layer
     *  and the weights of the resulting sub-layers after compression.
     */
    void ComputeOriginalAndCompressedMemory_(SVD_COMPRESS_TYPE mode, int rows, int cols, int k_w, int k_h,
                                             std::vector<unsigned int>& svd_ranks, size_t& original_size,
                                             size_t& reduced_size);

    /**
     * @brief Compute the original and reduced computations of a layer with a given set of ranks.
     * @param mode Compression mode (SVD/SSVD)
     * @param rows Number of rows of equivalent 2D matrix.
     * @param cols Number of columns of equivalent 2D matrix.
     * @param k_h Kernel height (for CONV layer)
     * @param k_w Kernel width (for CONV layer)
     * @param act_h Top activation height of the layer
     * @param act_w Top activation width of the layer
     * @param svd_ranks The vector of ranks with which layer is compressed.
     * @param original_size Original layer size
     * @param reduced_size Reduced_size layer size
     *  The method computes the original number of MACs occuring in a layer
     *  and the MACs of the resulting sub-layers after compression.
     */
    void ComputeOriginalAndCompressedMAC_(SVD_COMPRESS_TYPE mode, int rows, int cols, int k_h, int k_w, int act_h,
                                          int act_w, std::vector<unsigned int>& svd_ranks, size_t& original_size,
                                          size_t& reduced_size);

    /**
     * @brief Examine SVD ranks for compressibility.
     * @param mode Compression mode (SVD/SSVD)
     * @param rows Number of rows of equivalent 2D matrix.
     * @param cols Number of columns of equivalent 2D matrix.
     * @param k_h Kernel height (for CONV layer)
     * @param k_w Kernel width (for CONV layer)
     * @param svd_ranks The vector of ranks to be validated.
     *  The method examines if the given rank can actually
     *  compress a given layer or instead inflate it, based on
     *  the dimensions of the resulting sub-matrices.
     */
    bool ValidateRanksByMemory_(SVD_COMPRESS_TYPE mode, int rows, int cols, int k_h, int k_w,
                                std::vector<unsigned int>& svd_ranks);

    /**
     * @brief Examine SVD ranks for compressibility.
     * @param mode Compression mode (SVD/SSVD)
     * @param rows Number of rows of equivalent 2D matrix.
     * @param cols Number of columns of equivalent 2D matrix.
     * @param k_h Kernel height (for CONV layer)
     * @param k_w Kernel width (for CONV layer)
     * @param act_h Top activation height of the layer
     * @param act_w Top activation width of the layer
     * @param svd_ranks The vector of ranks to be validated.
     *  The method examines if the given rank can actually
     *  compress a given layer or instead inflate it, based on
     *  the MACs of the resulting sub-matrices.
     */
    bool ValidateRanksByMAC_(SVD_COMPRESS_TYPE mode, int rows, int cols, int k_h, int k_w, int act_h, int act_w,
                             std::vector<unsigned int>& svd_ranks);

    /**
     * @brief Estimate cost savings with SVD compression.
     * @param layer Iterator through LayerMap_
     * @param rank_index Index of the vector of candidate ranks.
     * @param useBestRanks Use LayerAttributes --> bestRanks to
     *  compute compression score (used during direct compression)
     *  instead of ranks indexed by rank_index (used in analysis phase).
     */
    std::tuple<size_t, size_t>
    EstimateReducedCost_(typename std::map<std::string, LayerAttributes<DTYPE>>::iterator layer, int rank_index,
                         bool useBestRanks);

    /**
     * @brief Generate a collection of potential ranks for compression.
     * @param layer Iterator through LayerMap_
     * @param rankPool Output: Collection of potential ranks.
     *  The method sets up a basic criterion of compressibility
     *  to screen potential ranks for compressing a given layer.
     */
    void FillRankPool_(typename std::map<std::string, LayerAttributes<DTYPE>>::iterator layer,
                       std::vector<std::vector<unsigned int>>& rankPool);

    /**
     * @brief Transpose a 4D matrix along its non-input modes.
     * {Applies to 4D convolution tensors stored in row-major format}
     * @param src Source tensor
     * @param dst Destination tensor
     * @param M source row dimension
     * @param N source coloumn dimension
     * @param k_h source filter height
     * @param k_w source filter width
     *  The method transposes a 4D convolution matrix
     *  along the non-input modes (row, col) while
     *  preserving kernel dimensions and shape intact
     *  in the destination matrix.
     */
    void Transpose_4DMatrix_(DTYPE* src, DTYPE* dst, int M, int N, int k_h, int k_w);

    /**
     * @brief Transpose the source layer weights.
     * @param layerAttrib Attributes (including weight blobs) of the layer.
     * @param transposedBlob The output transposed blob.
     */
    void TransposeSrcLayerWeights_(LayerAttributes<DTYPE>* layerAttrib, DTYPE* transposedBlob);

#ifdef USE_OPENCV

    /**
     * @brief Core SVD compression algorithm.
     * @param srcMat Weight matrix of original layer
     * @param layerA_Mat Weight matrix of split layer 1
     * @param layerB_Mat Weight matrix of split layer 2
     * @param r Rank to be used for compression.
     *  The method performs standard SVD on the source matrix
     *  to generate two sub-matrices, which are then sliced
     *  based on the input rank to achieve compression.
     */
    void SVDCompress_(cv::Mat& srcMat, cv::Mat& layerA_Mat, cv::Mat& layerB_Mat, unsigned int r);

    /**
     * @brief Truncate SVD sub-matrices based on rank selection.
     * @param U Matrix of Left singular vectors of source matrix after SVD.
     * @param W Diagonal Matrix of singular values of source matrix after SVD.
     * @param VT Matrix of Right singular vectors of source matrix after SVD.
     * @param layerA_Mat Weight matrix of sliced layer 1
     * @param layerB_Mat Weight matrix of sliced layer 2
     * @param r Rank to be used for compression.
     *  The method is similar to SVDCompress_ except that
     *  it truncates sub-matrices obtained by an SVD operation
     *  based on rank selections.
     */
    void TruncateMatrix_(cv::Mat& U, cv::Mat& W, cv::Mat& VT, cv::Mat& layerA_Mat, cv::Mat& layerB_Mat, unsigned int r);

#endif

    /**
     * @brief Estimate Tensor approximation residual (TAR).
     * @param layer Iterator through LayerMap_
     *  @param rankPool The complete collection collection of ranks
     *  with which to estimate TAR.
     *  @param TARMap Map of ranks with their residual values.
     *  The method determines the residual between the original weight matrix
     *  and an approximation obtained by reconstructing it from
     *  the components obtained by the SVD compression by evaluating
     *  the relative Frobenius norm of the difference between the two matrices.
     *  Generally higher rank values lead to lesser compression, thereby to
     *  lower residual values, and vice-versa.
     */
    void EstimateTAR_(typename std::map<std::string, LayerAttributes<DTYPE>>::iterator layer,
                      std::vector<std::vector<unsigned int>>& rankPool,
                      std::map<std::vector<unsigned int>, DTYPE>& TARMap);

    /**
     * @brief Pick a set of candidate ranks with varying TAR values.
     * @param TARMap Map of ranks with their residual values.
     * @candidate_ranks Output containing the determined set of ranks
     *  The method clusters the collection of ranks in the TARMap into
     *  a finite number of bins representing different relative TAR values
     *  scaled between (0, 1) and picks ranks representing each cluster.
     */
    void PickCandidateRanks_(std::map<std::vector<unsigned int>, DTYPE>& TARMap,
                             std::vector<std::vector<unsigned int>>& candidate_ranks);

    /**
     * @brief Access the class member 'BiasCorrection_'.
     */
    void SetBiasCorrection_(const std::string& layer_name, const std::vector<unsigned int>& ranks,
                            const std::vector<DTYPE>& bias_correction);

    /**
     * @brief Access the class member 'BiasCorrection_'.
     */
    std::vector<DTYPE> GetBiasCorrection_(const std::string& layer_name, const std::vector<unsigned int>& ranks);

#ifdef USE_OPENCV

    /**
     * @brief Perform mean shift correction (MSC) for a given layer and rank.
     * @param layer_name The name of the layer.
     * @param ranks The SVD or SSVD rank(s).
     * @param errorMat The error introduced by low-rank approximation.
     *
     * At a high level, the MSC method adds a correction term to the bias to
     * make up for a mean shift. When we approximate a layer by a low-rank
     * matrix, we introduce an error. The MSC method uses this error matrix, as
     * well as the mean input of a given layer, to compute a bias correction.
     *
     * Note that the bias correction will be stored internally (in
     * BiasCorrection_).
     *
     * Note: MSC relies on the mean input of a given layer. If the user didn't
     * provide this input mean, this method will compute a correction term of
     * zero.
     */
    void MSC_(const std::string& layer_name, const std::vector<unsigned int>& ranks, const cv::Mat& error_mat);

#endif

    // Map of layer names with their attributes.
    std::map<std::string, LayerAttributes<DTYPE>> LayerMap_;
    // Total network cost for MAC and Memory.
    size_t networkCost_Mem_;
    size_t networkCost_Mac_;
    // The bias corrections.
    // You should access this member through SetBiasCorrection_() and
    // GetBiasCorrection_().
    // This map is used for the MSC method, where we compute a bias correction
    // to make up for the error introduced in compression. We store the bias
    // correction for each layer, and for each rank setting.
    // The outer map contains pairs of layer names and the inner map. The inner
    // map contains pairs of rank settings and bias corrections.
    std::map<std::string, std::map<std::vector<unsigned int>, std::vector<DTYPE>>> BiasCorrection_;

    NETWORK_COST_METRIC metric_;
};

}   // End of namespace DlCompression
#endif   // SVDALGORITHM_H
