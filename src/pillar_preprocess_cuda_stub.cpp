// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "pcod_common/pillar_preprocess_cuda.hpp"

namespace pcod_common {

struct PillarPreprocessCudaContext::Impl {};

bool HasCudaPillarPreprocessSupport() { return false; }

PillarPreprocessCudaContext::PillarPreprocessCudaContext() : impl_(std::make_unique<Impl>()) {}

PillarPreprocessCudaContext::~PillarPreprocessCudaContext() = default;

PillarPreprocessCudaContext::PillarPreprocessCudaContext(PillarPreprocessCudaContext&&) noexcept = default;

PillarPreprocessCudaContext& PillarPreprocessCudaContext::operator=(PillarPreprocessCudaContext&&) noexcept = default;

bool PillarPreprocessCudaContext::isAvailable() const { return impl_ != nullptr && HasCudaPillarPreprocessSupport(); }

bool PillarPreprocessCudaContext::run(const PillarPreprocessPoint* points,
                                      std::int32_t num_points,
                                      const PillarPreprocessCudaConfig& config,
                                      const PillarPreprocessCudaOutputs& outputs,
                                      std::string* error_message) {
  (void)points;
  (void)num_points;
  (void)config;
  (void)outputs;
  if (error_message != nullptr) {
    *error_message =
        impl_ == nullptr ? "CUDA preprocessing context is not initialized" : "CUDA preprocessing is not available in this build";
  }
  return false;
}

bool PillarPreprocessCudaContext::runToDevice(const PillarPreprocessPoint* points,
                                              std::int32_t num_points,
                                              const PillarPreprocessCudaConfig& config,
                                              const PillarPreprocessCudaDeviceOutputs& outputs,
                                              std::string* error_message) {
  (void)points;
  (void)num_points;
  (void)config;
  (void)outputs;
  if (error_message != nullptr) {
    *error_message =
        impl_ == nullptr ? "CUDA preprocessing context is not initialized" : "CUDA preprocessing is not available in this build";
  }
  return false;
}

}  // namespace pcod_common
