import torch
import time

def check_cuda():
    if torch.cuda.is_available():
        print("success")
        return True
    else:
        print("failure")
        return False

def perform_matrix_multiplication():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    success_count = 0

    for i in range(1, 11):
        try:
            size = i * 1000  # Gradually increase the size of the matrix
            A = torch.randn(size, size, device=device)
            B = torch.randn(size, size, device=device)

            start_time = time.time()
            C = torch.matmul(A, B)
            torch.cuda.synchronize()  # Wait for the operation to complete
            end_time = time.time()

            print(f"Loop {i}: success (Matrix size: {size}x{size}, Time taken: {end_time - start_time:.4f} seconds)")
            success_count += 1
        except Exception as e:
            print(f"Loop {i}: failure ({str(e)})")

    if success_count == 10:
        print("final success")
    else:
        print("Some operations failed. Check the output above for details.")

if __name__ == "__main__":
    if check_cuda():
        perform_matrix_multiplication()
