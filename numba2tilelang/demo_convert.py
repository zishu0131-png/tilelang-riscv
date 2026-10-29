"""Print TileLang source for a small, locally authored CUDA Python example.

This only parses source and generates text; it does not import or run the kernel.
"""
from convert import convert

SOURCE = '''from numba import cuda

@cuda.jit
def add(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = A[i, j] + B[i, j]
'''

if __name__ == '__main__':
    print(convert(SOURCE))
