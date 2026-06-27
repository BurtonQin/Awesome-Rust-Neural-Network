# Awesome-Rust-Neural-Network
A curated collection of Rust projects related to neural networks, designed to complement "[Are We Learning Yet](https://www.arewelearningyet.com)". I plan to submit libraries to "Are We Learning Yet" that are included in this project but currently missing from it. 

This repository aims to provide a comprehensive overview of the Rust neural network ecosystem to identify the currently missing foundational infrastructure. Thus, it focuses more on the features, backends, and main dependencies of the Rust-based neural network projects. Note that the features, backends, and main dependencies are based on my own search and understanding, and may be incomplete or even incorrect.

## Table of contents

- [Awesome-Rust-Neural-Network](#awesome-rust-neural-network)
  - [Table of contents](#table-of-contents)
  - [Neural Networks](#neural-networks)
  - [Machine Learning](#machine-learning)
  - [GPU Computing](#gpu-computing)
  - [Miscellaneous](#miscellaneous)

## Neural Networks


| Name | Description | Features | Backends | Main Deps | Last Commit Time |
| -----| ----------- | ---------- | ---------- | ---------- | ---------- |
| [candle](https://github.com/huggingface/candle) | Minimalist ML framework for Rust | Like PyTorch, Training, Various Models | CPU, CUDA, CUDA NCCL, WASM | gemm, intel-mkl-src, cudarc, metal, accelerate-src | 2026-06-26 |
| [burn](https://github.com/tracel-ai/burn) | Burn is a new comprehensive dynamic Deep Learning Framework built using Rust with extreme flexibility, compute efficiency and portability as its primary goals. | Various backends, Kernel Fusion, Training, Various Models, ONNX | WGPU, Candle, Torch, Ndarray, Remote | matrixmultiply, blas-src, libm, openblas-src, ndarray, candle-core, cubecl, cudarc, tch  | 2026-06-26 |
| [dfdx](https://github.com/coreylowman/dfdx) | Deep learning in Rust, with shape checked tensors and neural networks | Compile-time Checking | CPU, CUDA, WGPU | gemm, cudarc, wgpu | 2024-01-25 |
| [luminal](https://github.com/jafioti/luminal) | Deep learning at the speed of light | Static Computation Graph, RISC-style arch, Kernel Fusion, Training | CPU, CUDA, Metal | matrixmultiply, cudarc, metal-rs | 2026-06-22 |
| [autograph](https://github.com/charles-r-earp/autograph) | A machine learning library for Rust | GPGPU kernels implemented with krnl | CPU, Vulkan | krnl, ndarray | 2024-08-19 |
| [unda](https://github.com/unda-ml/unda) | General purpose machine learning crate | Compile to XLA | XLA | xla-rs | 2024-06-19 |
| [custos](https://github.com/elftausend/custos) | A minimal OpenCL, CUDA, Vulkan and host CPU array manipulation engine / framework | Array Manipulation, AutoDiff, Lazy Execution | CPU, OpenCL, CUDA, Vulkan, NNAPI | min-cl, libm, ash, naga, nnapi | 2025-08-21 |
| [zyx](https://github.com/zk4x/zyx) | Tensor library for machine learning | Lazy Execution, AutoDiff | CUDA, OpenCL, WGPU | wgpu, vulkano, manual bindings to CUDA, OpenCL, and HSA | 2026-06-26 |
| [zenu](https://github.com/bokutotu/zenu) | A Deep Learning Framework Written in Rust | Training, AutoDiff | CPU, CUDA | cblas, openblas-src, manual binding to CUDA | 2024-12-30 |
| [maidenx](https://github.com/miniex/maidenx) | A lightweight and fast AI framework in Rust focused on simplicity and performance | Educational focus, Mirror PyTorch's arch| CPU, CUDA | - | 2025-10-19 |
| [ort](https://github.com/pykeio/ort) | Fast ML inference & training for ONNX models in Rust | ONNX, Various Backends | CUDA, TensorRT, OpenVINO, oneDNN, DirectML, QNN, CoreML, ACL, TVM, CANN, etc. | ort-sys: Unsafe Rust bindings for ONNX Runtime 1.20  | 2026-06-12 |
| [tract](https://github.com/sonos/tract) | Tiny, no-nonsense, self-contained, Tensorflow and ONNX inference | ONNX, Tensorflow | CPU, Metal | accelerate-src, blis-src, cblas, metal, ndarray, openblas-src, tensorflow, tflitec | 2026-06-27 |
| [Kyanite](https://github.com/KarelPeeters/Kyanite) | A neural network inference library, written in Rust | ONNX, Graph IR | CPU, CUDA | manual binding to CUDA | 2026-03-15 |
| [rten](https://github.com/robertknight/rten) | ONNX neural network inference engine | ONNX | CPU | rayon | 2026-06-27 |
| [oxide-rs](https://github.com/theawakener0/oxide-rs) | High-performance LLM inference engine inspired by llama.cpp. | LLM Inference, Quantization | CPU | llama.cpp | 2026-03-16 |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Blazingly fast LLM inference | LLM inference, safetensors, Quantization | CPU, CUDA, Metal | mkl, candle, metal, accelerate | 2026-06-27 |
| [InfiniLM](https://github.com/InfiniTensor/InfiniLM) | A handwriting transformer model project developed from YdrMaster/llama2.rs | LLM Inference, Multiple backends supported | CPU, CUDA, OpenCL, Ascend, etc. | operators | 2026-06-26 |
| [operators](https://github.com/YdrMaster/operators-rs) | Multi-hardware support operator library | Multi-hardware | CPU, CUDA, OpenCL, Ascend, Cambricon | clrt, infinirt, cuda-driver | 2025-02-19 |
| [crabml](https://github.com/crabml/crabml) | a fast cross platform AI inference engine using Rust and WebGPU | LLM Inference, mmap, Quantization | CPU, WGPU | vulkano, wgpu | 2025-01-04 |
| [diffusion-rs](https://github.com/EricLBuehler/diffusion-rs) | Blazingly fast inference of diffusion models | Diffusion, Quantization, DDUF, Offloading | CPU, CUDA, Metal, etc. | cudarc, intel-mkl-src, accelerate-sr, metal, gemm | 2025-04-01 |
| [mmnn](https://github.com/GrgoMariani/mmnnrust) | rust-based bash-cli for Neural Network propagation/backpropagation | bash-cli, json-config | - | - | 2025-06-28 |

## Machine Learning

| Name | Description | Features | Backends | Main Deps | Last Commit Time |
| -----| ----------- | ---------- | ---------- | ---------- | ---------- |
| [linfa](https://github.com/rust-ml/linfa) | A Rust machine learning framework | like scikit-learn | CPU | ndarray, sprs | 2026-05-30 |
| [jams-rs](https://github.com/gagansingh894/jams-rs) | Rust based model serving solution for popular machine learning frameworks | Model Store, PyTorch & Tenosrflow Models, Tree Models | PyTorch, TensorFlow, Catboost, LightGBM | tensorflow, tch, catboost-rs, lgbm, ndarray | 2025-05-30 |
| [bullet](https://github.com/jw1912/bullet) | Specialised ML Library | Domain-specific, NNUE-style networks, chess engines | cudarc | - | 2026-06-22 |
| [redstone-ml](https://github.com/BhavyeMathur/redstone-ml) | High-performance Machine Learning, Dynamic Auto-Differentiation and Tensor Algebra crate for Rust | AutoDiff, NdArrray | CPU, GPU | - | 2025-06-08 |
| [RustyML](https://github.com/SomeB1oody/RustyML) | A high-performance machine learning library in pure Rust, offering statistical utilities, ML algorithms and neural networks. | - | CPU | ndarray, rayon | 2026-06-25 |

## GPU Computing

| Name | Description | Features | Backends | Main Deps | Last Commit Time |
| -----| ----------- | ---------- | ---------- | ---------- | ---------- |
| [cubecl](https://github.com/tracel-ai/cubecl) | Multi-platform high-performance compute language extension for Rust | Kernels in Rust, CubeIR | WGPU, CUDA, HIP |  cudarc, cubecl-hip-sys, ash, cubecl-spirv, wgpu | 2026-06-25 |
| [krnl](https://github.com/charles-r-earp/krnl) | Safe, portable, high performance compute (GPGPU) kernels | Kernels in Rust | Vulkan | rsprv, vulkano, ash | 2025-10-20 |
| [EnzymeAD](https://github.com/EnzymeAD/rust) | A rust fork to work towards Enzyme integration | AutoDiff on LLVM |  LLVM | Enzyme | 2024-11-25 |
| [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA)| Ecosystem of libraries and tools for writing and executing fast GPU code fully in Rust | Compiling Rust to PTX | CUDA | Binding | 2026-04-29 | 
| [Rust-GPU](https://github.com/Rust-GPU/rust-gpu)| Making Rust a first-class language and ecosystem for GPU shaders | Compiling Rust to SPIR-V | SPIR-V | Binding | 2026-06-23 | 
| [cudarc](https://github.com/chelsea0x3b/cudarc)| Safe rust wrapper around CUDA toolkit | Mixture of safe and sys APIs | CUDA | Binding | 2026-06-19 | 
| [async-cuda](https://github.com/oddity-ai/async-cuda) | Asynchronous CUDA for Rust | Async, NPP | CUDA | Binding | 2026-06-02 |
| [async-tensorrt](https://github.com/oddity-ai/async-tensorrt) | Asynchronous TensorRT for Rust | Async, TensorRT | CUDA | async-cuda, Binding | 2026-06-02 |
| [cuda-rs](https://github.com/vivym/cuda-rs) | Rust wrapper for CUDA Driver (libcuda.so) and Runtime (libcudart.so) APIs | - | CUDA | Binding | 2024-01-01 |
| [tensorrt-rs](https://github.com/vivym/tensorrt-rs)| Rust wrapper to NVIDIA TensorRT | TensorRT | CUDA | cuda-rs, Binding | 2024-01-03 | 
| [oxicuda](https://github.com/cool-japan/oxicuda) | OxiCUDA replaces the entire NVIDIA CUDA Toolkit software stack with type-safe, memory-safe Rust code | CUDA driver | CUDA | - | 2026-06-25 |
| [cuda-oxide](https://github.com/NVlabs/cuda-oxide) | An experimental Rust-to-CUDA compiler that lets you write (SIMT) GPU kernels in safe(ish), idiomatic Rust | Rust to PTX | CUDA | - | 2026-06-27 |
| [Rurix](https://github.com/qwasg/Rurix) | GPU programming language and toolchain written in Rust | CUDA first, Windows first | CUDA driver | - | 2026-06-25 |


## Miscellaneous

- [gpu-allocator](https://github.com/Traverse-Research/gpu-allocator), GPU memory allocator for Vulkan, DirectX 12 and Metal. Written in pure Rust.
- [ZLUDA](https://github.com/vosen/ZLUDA), CUDA on non-NVIDIA GPUs.
- [adk-rust](https://github.com/zavora-ai/adk-rust), Production-ready AI agent development kit for Rust.
- [rust-norion](https://github.com/yanghao1143/rust-norion), Rust Noiron/Norion self-evolving local LLM runtime engine for non-commercial research deployment.
