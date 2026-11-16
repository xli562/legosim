import os
import sys
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
import numpy as np

load_dotenv()
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
PYUTILS_RDIR = Path(os.getenv('PYUTILS_RDIR'))
sys.path.insert(0, str(PROJECT_ROOT / PYUTILS_RDIR))
from xlogging import logger

M = 150
K = 60
N = 200
RNG_P1 = 7
RNG_P2 = 13
SEED_A = 321
SEED_B = 37
element_t = np.int64

run_output_rdir = Path('run_outputs/gemm_2dmesh.EaEyRy')
output_cpu_thread_dirname = Path('proc_r3_p1_t0')


def matPrng(pos:int, colCnt:int, p1:int, p2:int, seed:int) -> int:
    """ Python version of the `matPrng` function in `cpu.cpp` """
    i = pos // colCnt
    j = pos % colCnt
    return (i*p1 + j*p2 + seed) % 51

def gen_input_matrix(mat_name:Literal['A','B']) -> np.ndarray:
    """ Generates matA and matB exactly as in `cpu.cpp` """
    row_cnt = M if mat_name == 'A' else K
    col_cnt = K if mat_name == 'A' else N
    seed = SEED_A if mat_name == 'A' else SEED_B
    retmat = np.ndarray((row_cnt, col_cnt), dtype=element_t)
    for i in range(row_cnt * col_cnt):
        retmat.flat[i] = matPrng(i, col_cnt, RNG_P1, RNG_P2, seed)
    return retmat

def verify_output_matrix(matA, matB, matC_path) -> bool:
    """ Calculates matA@matB and checks matC against the result """
    matC = np.fromfile(matC_path, dtype='<i8')
    if matC.size != M * N:
        logger.error(f'Output matrix at {matC_path} has size {matC.size}, '
                     f'{M * N} expected')
        return False
    matC = matC.reshape(M, N)
    return True if np.all(matA @ matB == matC) else False

def main():
    matA = gen_input_matrix('A')
    matB = gen_input_matrix('B')
    matC_path = PROJECT_ROOT / run_output_rdir \
                / output_cpu_thread_dirname / 'matC.bin'
    if verify_output_matrix(matA, matB, matC_path):
        logger.debug('Checked AB == C')
    else:
        logger.warning(f'Incorrect output matrix at {matC_path}')

if __name__ == '__main__':
    main()
