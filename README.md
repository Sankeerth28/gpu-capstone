## CUDA Matrix Multiplication Capstone

This project benchmarks CPU vs GPU matrix multiplication using NVIDIA cuBLAS.

### Environment
- Google Colab GPU (Tesla T4)
- CUDA 12.4
- cuBLAS

### Build
nvcc matmul.cu -lcublas -o matmul

### Run
./matmul

### Results
See timing.csv for performance comparison.
