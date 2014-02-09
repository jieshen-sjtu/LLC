/*
 * LLC_Encoder.hpp
 *
 *  Created on: Feb 2, 2014
 *      Author: jieshen
 */

#ifndef __LLC_Encoder_LLC_ENCODER_HPP__
#define __LLC_Encoder_LLC_ENCODER_HPP__

#include <stdint.h>
#include <cmath>
#include <vl/kdtree.h>
#include <boost/shared_ptr.hpp>

namespace jieshen
{
  using boost::shared_ptr;

  class LLC_Encoder
  {
    // constructor and destructor
  public:
    LLC_Encoder();
    LLC_Encoder(const shared_ptr<float>& base, const uint32_t dim,
                const uint32_t num_base);
    ~LLC_Encoder();

    // must call this function before encoder!!!
    void SetUp();
    void clear();

  public:
    void Encoder(const shared_ptr<float>& X, const uint32_t dim,
                 const uint32_t num_frame, shared_ptr<float>& codes);

  private:
    void init_with_default_parameter();
    void clear_data();

  public:
    // setting and accessing
    void set_base(const shared_ptr<float>& base, const uint32_t dim,
                  const uint32_t num_base);

    inline void set_thrd_method(const VlKDTreeThresholdingMethod method)
    {
      if (method == thrd_method_)
        return;
      thrd_method_ = method;
      has_setup_ = false;
    }
    inline void set_dist_method(const VlVectorComparisonType method)
    {
      if (method == dist_method_)
        return;
      dist_method_ = method;
      has_setup_ = false;
    }
    inline void set_num_tree(const uint32_t num_tree)
    {
      if (num_tree == num_tree_)
        return;
      num_tree_ = num_tree;
      has_setup_ = false;
    }
    inline void set_num_knn(const uint32_t num_knn)
    {
      if (num_knn == num_knn_)
        return;
      num_knn_ = num_knn;
      has_setup_ = false;
    }
    inline void set_max_comp(const uint32_t max_comp)
    {
      if (max_comp == max_comp_)
        return;
      max_comp_ = max_comp;
      has_setup_ = false;
    }
    inline void set_beta(const float beta)
    {
      if (std::abs(beta - beta_) < 1e-10)
        return;
      beta_ = beta;
      has_setup_ = false;
    }

    inline const float* get_base() const
    {
      return base_;
    }
    inline const uint32_t get_dim() const
    {
      return dim_;
    }
    inline const uint32_t get_num_base() const
    {
      return num_base_;
    }
    inline const VlKDTreeThresholdingMethod get_thrd_method() const
    {
      return thrd_method_;
    }
    inline const VlVectorComparisonType get_dist_method() const
    {
      return dist_method_;
    }
    inline const uint32_t get_num_tree() const
    {
      return num_tree_;
    }
    inline const uint32_t get_num_knn() const
    {
      return num_knn_;
    }
    inline const uint32_t get_max_comp() const
    {
      return max_comp_;
    }
    inline const float get_beta() const
    {
      return beta_;
    }

  public:
    enum
    {
      DEFAULT_NUM_TREE = 1, DEFAULT_NUM_KNN = 5, DEFAULT_MAX_COMP = 500,
    };
#define DEFAULT_THRD_METHOD VL_KDTREE_MEDIAN
#define DEFAULT_DIST_METHOD VlDistanceL2
#define DEFAULT_BETA 1e-4

  private:
    // base data
    float* base_; // the actual base, get from shared_ptr
    uint32_t dim_;
    uint32_t num_base_;

    // kd-forest data
    VlKDForest* kdforest_;
    VlKDTreeThresholdingMethod thrd_method_;
    VlVectorComparisonType dist_method_;
    uint32_t num_tree_;
    uint32_t num_knn_;
    uint32_t max_comp_;

    // LLC parameter
    float beta_;

    // tag
    bool has_setup_;
  };
}

#endif /* __LLC_Encoder_LLC_ENCODER_HPP__ */
