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

bool PillarPreprocessCudaContext::isAvailable() const { return false; }

bool PillarPreprocessCudaContext::run(const PillarPreprocessPoint*, std::int32_t, const PillarPreprocessCudaConfig&,
                                      const PillarPreprocessCudaOutputs&, std::string* error_message) {
  if (error_message != nullptr) {
    *error_message = "CUDA preprocessing is not available in this build";
  }
  return false;
}

bool PillarPreprocessCudaContext::runToDevice(const PillarPreprocessPoint*, std::int32_t,
                                              const PillarPreprocessCudaConfig&,
                                              const PillarPreprocessCudaDeviceOutputs&,
                                              std::string* error_message) {
  if (error_message != nullptr) {
    *error_message = "CUDA preprocessing is not available in this build";
  }
  return false;
}

}  // namespace pcod_common
