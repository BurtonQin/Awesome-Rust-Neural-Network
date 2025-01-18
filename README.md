# Awesome-Rust-Neural-Network
A curated collection of Rust projects related to neural networks, designed to complement "[Are We Learning Yet](https://www.arewelearningyet.com)". I plan to submit libraries to "Are We Learning Yet" that are included in this project but currently missing from it. 

This repository aims to provide a comprehensive overview of the Rust neural network ecosystem to identify the currently missing foundational infrastructure. Thus, it focuses more on the features, backends, and main dependencies of the Rust-based neural network projects. Note that the features, backends, and main dependencies are based on my own search and understanding, and may be incomplete or even incorrect.

## Table of contents

- [Awesome-Rust-Neural-Network](#awesome-rust-neural-network)
  - [Table of contents](#table-of-contents)
  - [Neural Networks](#neural-networks)
  - [GPU Computing](#gpu-computing)
  - [Miscellaneous](#miscellaneous)

## Neural Networks


| Name | Description | Features | Backends | Main Deps | Last Commit Time |
| -----| ----------- | ---------- | ---------- | ---------- | ---------- |
| [candle](https://github.com/huggingface/candle) | Minimalist ML framework for Rust | Like PyTorch, Training, Various Models | CPU, CUDA, CUDA NCCL, WASM | gemm, intel-mkl-src, cudarc, metal, accelerate-src | 2025-01-16 |
| [burn](https://github.com/tracel-ai/burn) | Burn is a new comprehensive dynamic Deep Learning Framework built using Rust with extreme flexibility, compute efficiency and portability as its primary goals. | Various backends, Kernel Fusion, Training, Various Models, ONNX | WGPU, Candle, Torch, Ndarray | matrixmultiply, blas-src, libm, openblas-src, ndarray, candle-core, cubecl, cudarc, tch  | 2025-01-16 |
| [dfdx](https://github.com/coreylowman/dfdx) | Deep learning in Rust, with shape checked tensors and neural networks | Compile-time Checking | CPU, CUDA, WGPU | gemm, cudarc, wgpu | 2024-01-25 |
| [luminal](https://github.com/jafioti/luminal) | Deep learning at the speed of light | Static Computation Graph, RISC-style arch, Kernel Fusion, Training | CPU, CUDA, Metal | matrixmultiply, cudarc, metal-rs | 2024-10-09 |
| [autograph](https://github.com/charles-r-earp/autograph) | A machine learning library for Rust | GPGPU kernels implemented with krnl | CPU, Vulkan | krnl, ndarray | 2024-08-19 |
| [unda](https://github.com/unda-ml/unda) | General purpose machine learning crate | Compile to XLA | XLA | xla-rs | 2024-06-19 |
| [custos](https://github.com/elftausend/custos) | A minimal OpenCL, CUDA, Vulkan and host CPU array manipulation engine / framework | Array Manipulation, AutoDiff, Lazy Execution | CPU, OpenCL, CUDA, Vulkan, NNAPI | min-cl, libm, ash, naga, nnapi | 2024-10-13 |
| [zyx](https://github.com/zk4x/zyx) | Tensor library for machine learning | Lazy Execution, AutoDiff | CUDA, OpenCL, WGPU | wgpu, vulano, manual bindings to CUDA, OpenCL, and HSA | 2025-01-13 |
| [zenu](https://github.com/bokutotu/zenu) | A Deep Learning Framework Written in Rust | Training, AutoDiff | CPU, CUDA | cblas, openblas-src, manual binding to CUDA | 2024-12-30 |
| [Kyanite](https://github.com/KarelPeeters/Kyanite) | A neural network inference library, written in Rust | ONNX, Graph IR | CPU, CUDA | manual binding to CUDA | 2024-07-13 |
| [tract](https://github.com/sonos/tract) | Tiny, no-nonsense, self-contained, Tensorflow and ONNX inference | ONNX, Tensorflow | CPU, Metal | accelerate-src, blis-src, cblas, metal, ndarray, openblas-src, tensorflow, tflitec | 2025-01-17 |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Blazingly fast LLM inference | LLM inference, safetensors, Quantization | CPU, CUDA, Metal | mkl, candle, metal, accelerate | 2025-01-18 |
| [InfiniLM](https://github.com/InfiniTensor/InfiniLM) | A handwriting transformer model project developed from YdrMaster/llama2.rs | LLM Inference, Multiple backends supported | CPU, CUDA, OpenCL, Ascend, etc. | operators | 2025-01-15 |
| [operators](https://github.com/YdrMaster/operators-rs) | Multi-hardware support operator library | Multi-hardware | CPU, CUDA, OpenCL, Ascend, Cambricon | clrt, infinirt, cuda-driver | 2025-01-15 |
| [crabml](https://github.com/crabml/crabml) | a fast cross platform AI inference engine using Rust and WebGPU | LLM Inference, mmap, Quantization | CPU, WGPU | vulkano, wgpu | 2025-01-04 |

## GPU Computing


| Name | Description | Features | Backends | Main Deps | Last Commit Time |
| -----| ----------- | ---------- | ---------- | ---------- | ---------- |
| [cubecl](https://github.com/tracel-ai/cubecl) | Multi-platform high-performance compute language extension for Rust | Kernels in Rust, CubeIR | WGPU, CUDA, HIP |  cudarc, cubecl-hip-sys, ash, cubecl-spirv, wgpu | 2025-01-17 |
| [krnl](https://github.com/charles-r-earp/krnl) | Safe, portable, high performance compute (GPGPU) kernels | Kernels in Rust | Vulkan | rsprv, vulkano, ash | 2024-05-28 |
| [async-cuda](https://github.com/oddity-ai/async-cuda) | Asynchronous CUDA for Rust | Async, NPP | CUDA | Binding | 2024-11-04 |
| [async-tensorrt](https://github.com/oddity-ai/async-tensorrt) | Asynchronous TensorRT for Rust | Async, TensorRT | CUDA | async-cuda, Binding | 2024-11-04 |
| [cuda-rs](https://github.com/vivym/cuda-rs) | Rust wrapper for CUDA Driver (libcuda.so) and Runtime (libcudart.so) APIs | - | CUDA | Binding | 2024-01-01 |
| [tensorrt-rs](https://github.com/vivym/tensorrt-rs)| Rust wrapper to NVIDIA TensorRT | TensorRT | CUDA | cuda-rs, Binding | 2024-01-03 | 

## Miscellaneous

- [gpu-allocator](https://github.com/Traverse-Research/gpu-allocator), GPU memory allocator for Vulkan, DirectX 12 and Metal. Written in pure Rust.
- [ZLUDA](https://github.com/vosen/ZLUDA), CUDA on non-NVIDIA GPUs.
