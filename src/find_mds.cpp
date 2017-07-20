#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <queue>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include "wmmintrin.h"
#include <smmintrin.h>

#define NB_INPUTS 4
#define NB_REGISTERS 5
#define KEEP_INPUTS 0

#define XOR_WEIGHT 2
#define MUL_WEIGHT 1
#define CPY_WEIGHT 0

#define COMPUTE_ID_FIRST

enum op_name {XOR=0, MUL=1, CPY=2};

typedef struct {
    enum op_name type;
    char from, to;
} algo_op;

typedef struct ALGO_STATE {
    // uint32_t is considering we won't have more than about 20 multiplications, therefore we need at least 20 bits not to overflow.
    uint32_t ** branch_vals;
    char weight, weight_to_MDS;
    struct ALGO_STATE * pred;
    algo_op * op;
} algo_state;

struct AlgoHasher {
    std::size_t operator()(uint32_t ** const arr) const {
        std::size_t seed = 0;
        for (int i=0; i<NB_REGISTERS; i++) {
            for (int j=0; j<NB_INPUTS; j++) {
                //seed ^= arr[i][j];
                boost::hash_combine(seed, arr[i][j]);
            }
        }
        return seed;
    }
};

struct HashComparator {
  bool operator()(uint32_t ** const a, uint32_t ** const b) const {
    for (int i=0; i<NB_REGISTERS; i++) {
        for (int j=0; j<NB_INPUTS; j++) {
            if (a[i][j] != b[i][j]) {
                return false;
            }
        }
    }
    return true;
  }
};

struct QueueComparator {
    bool operator()(const algo_state * a, const algo_state * b) const {
        return (a->weight + a->weight_to_MDS) > (b->weight + b->weight_to_MDS);
    }
};

void print_state (algo_state * current_state, int nb_scanned, int nb_tested, int print_all, int print_pred) {
    int depth = 0;
    algo_state * depth_finder;
    for (depth_finder=current_state; depth_finder->pred != NULL; depth_finder=depth_finder->pred, depth++);
    if (print_all) {
        printf ("Number of distinct ids (weight=%3d, depth=%3d) : %" PRIu32 "/%" PRIu32 "(1/%.1f)\n", current_state->weight, depth, nb_scanned, nb_tested, (nb_scanned?(float)nb_tested/nb_scanned:0));
    }
        
    if (print_all) {
        int i, input_n;
        printf ("Current state\n");
        for (i=0; i<NB_REGISTERS; i++) {
            for (input_n=0; input_n<NB_INPUTS; input_n++)
                printf ("%d ", current_state->branch_vals[i][input_n]);
            printf ("\n");
        }
        if (current_state->op != NULL)
            printf ("Current state op : %c (%d, %d)\n", (current_state->op->type==XOR?'x':(current_state->op->type==MUL?'m':'c')), current_state->op->from, current_state->op->to);
    }
     if (print_pred) {
        printf ("Printing preds\n#############################\n");
        for (depth_finder=current_state; depth_finder->pred != NULL; depth_finder=depth_finder->pred) {
            printf ("%c (%d, %d)\n", (depth_finder->op->type==XOR?'x':(depth_finder->op->type==MUL?'m':'c')), depth_finder->op->from, depth_finder->op->to);
            if (print_all)
                print_state(depth_finder->pred, nb_scanned, nb_tested, false, false);
        }
     }
     printf ("End of state\n");
}

bool test_injective (uint32_t ** M, char selected_outputs[NB_INPUTS]) {
    /* Note that this function assumes that we have a N x 4 matrix M, and we select 4 lines, yielding a 4 x 4 matrix. 
     * This test only happens after adding a copy operation (only operation that can change injectivity).
     * In that case, the matrix gets 2 identical lines.
     * We consider that NB_REGISTERS = NB_INPUTS + 1, therefore we are left with a square matrix.
     * We simply compute its determinant and return if it is 0 or not.
     */
    
    __m128i dim2det[NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS]; /* Not optimal in terms of memory, we could do with a [4][3][4][3] since the lines and columns must be different. */
    __m128i dim3det[NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS]; /* Not optimal in terms of memory. Again, lines and columns different, but also we could just store the eliminated lines and columns. */
    
    
    __m128i MM[NB_INPUTS][NB_INPUTS];
    
    int line1, line2, line3, column1, column2, column3;
    
    /* Dimension 1 determinants. */
    for (line1=0; line1<NB_INPUTS; line1++) {
        for (column1=0; column1<NB_INPUTS; column1++) {
            MM[line1][column1] = _mm_set_epi32(0, 0, 0, M[selected_outputs[line1]][column1]);
            if (NB_INPUTS==1 && M[selected_outputs[line1]][column1] == 0) // Wrong : tests if MDS
                return false;
        }
    }
    
    if (NB_INPUTS > 1) {
        /* Dimension 2 determinants. Not considering the bottom lines, we will use them to build the dimension 3 determinants only.*/
        for (column1=0; column1<NB_INPUTS; column1++) {
            for (column2=column1+1; column2<NB_INPUTS; column2++) {
                dim2det[0][1][column1][column2] = _mm_xor_si128(_mm_clmulepi64_si128(MM[0][column1], MM[1][column2], 0x00) , \
                                                                                _mm_clmulepi64_si128(MM[0][column2], MM[1][column1], 0x00));
                if (NB_INPUTS==2 && _mm_testz_si128(dim2det[0][1][column1][column2], dim2det[0][1][column1][column2])) // Test if det is zero.
                    return false;
            }
        }
        
        if (NB_INPUTS > 2) {
            /* Dimension 3 determinants. Not considering the top line, we will use it to build the dimension 4 determinant only.*/
            for (column1=0; column1<NB_INPUTS; column1++) {
                for (column2=column1+1; column2<NB_INPUTS; column2++) {
                    for (column3=column2+1; column3<NB_INPUTS; column3++) {
                        dim3det[0][1][2][column1][column2][column3] = _mm_xor_si128(
                                                                                    _mm_xor_si128(_mm_clmulepi64_si128(MM[2][column1], dim2det[0][1][column2][column3], 0x00), \
                                                                                                        _mm_clmulepi64_si128(MM[2][column2], dim2det[0][1][column1][column3], 0x00)), \
                                                                                    _mm_clmulepi64_si128(MM[2][column3], dim2det[0][1][column1][column2], 0x00));
                        if (NB_INPUTS==3 && _mm_testz_si128(dim3det[0][1][2][column1][column2][column3], dim3det[0][1][2][column1][column2][column3])) // Test if det is zero.
                            return false;
                    }
                }
            }
            
            if (NB_INPUTS > 3) {
                /* Dimension 4 determinant != 0. */
                /* Multiplying a 32-bit word with a 96-bit word takes work.
                * We use: Let u the 32-bit word, v||w the 96-bit word, with v on 32 bits, w on 64 bits.
                * r = (uxw) ^ [(uxv)<<64].
                */
                __m128i det4 = _mm_setzero_si128();
                __m64 zero64 = _mm_setzero_si64();
                __m128i v, w;
                v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[0][1][2][1][2][3], 1));
                w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[0][1][2][1][2][3], 0));
                __m128i mul1 = _mm_clmulepi64_si128(MM[3][0], w, 0x00);
                __m128i mul2 = _mm_clmulepi64_si128(MM[3][0], v, 0x00);
                mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
                det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
                
                v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[0][1][2][0][2][3], 1));
                w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[0][1][2][0][2][3], 0));
                mul1 = _mm_clmulepi64_si128(MM[3][1], w, 0x00);
                mul2 = _mm_clmulepi64_si128(MM[3][1], v, 0x00);
                mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
                det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
                
                v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[0][1][2][0][1][3], 1));
                w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[0][1][2][0][1][3], 0));
                mul1 = _mm_clmulepi64_si128(MM[3][2], w, 0x00);
                mul2 = _mm_clmulepi64_si128(MM[3][2], v, 0x00);
                mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
                det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
                
                v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[0][1][2][0][1][2], 1));
                w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[0][1][2][0][1][2], 0));
                mul1 = _mm_clmulepi64_si128(MM[3][3], w, 0x00);
                mul2 = _mm_clmulepi64_si128(MM[3][3], v, 0x00);
                mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
                det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
                
                if (_mm_testz_si128(det4, det4))
                    return false;
            }
        }
    }
    return true;
}

char order_permutations[6][120][5] = {
    {{}},
    {{0}},
    {{0, 1}, {1, 0}},
    {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}},
    {{0, 1, 2, 3},
    {0, 1, 3, 2},
    {0, 3, 1, 2},
    {0, 3, 2, 1},
    {0, 2, 1, 3},
    {0, 2, 3, 1},
    {1, 2, 3, 0},
    {1, 2, 0, 3},
    {1, 3, 2, 0},
    {1, 3, 0, 2},
    {1, 0, 3, 2},
    {1, 0, 2, 3},
    {2, 0, 1, 3},
    {2, 0, 3, 1},
    {2, 1, 0, 3},
    {2, 1, 3, 0},
    {2, 3, 0, 1},
    {2, 3, 1, 0},
    {3, 0, 1, 2},
    {3, 0, 2, 1},
    {3, 1, 0, 2},
    {3, 1, 2, 0},
    {3, 2, 0, 1},
    {3, 2, 1, 0}},
    {{0, 1, 2, 3, 4},
    {0, 1, 3, 2, 4},
    {0, 3, 1, 2, 4},
    {0, 3, 2, 1, 4},
    {0, 2, 1, 3, 4},
    {0, 2, 3, 1, 4},
    {1, 2, 3, 0, 4},
    {1, 2, 0, 3, 4},
    {1, 3, 2, 0, 4},
    {1, 3, 0, 2, 4},
    {1, 0, 3, 2, 4},
    {1, 0, 2, 3, 4},
    {2, 0, 1, 3, 4},
    {2, 0, 3, 1, 4},
    {2, 1, 0, 3, 4},
    {2, 1, 3, 0, 4},
    {2, 3, 0, 1, 4},
    {2, 3, 1, 0, 4},
    {3, 0, 1, 2, 4},
    {3, 0, 2, 1, 4},
    {3, 1, 0, 2, 4},
    {3, 1, 2, 0, 4},
    {3, 2, 0, 1, 4},
    {3, 2, 1, 0, 4},
    {0, 1, 2, 4, 3},
    {0, 1, 3, 4, 2},
    {0, 3, 1, 4, 2},
    {0, 3, 2, 4, 1},
    {0, 2, 1, 4, 3},
    {0, 2, 3, 4, 1},
    {1, 2, 3, 4, 0},
    {1, 2, 0, 4, 3},
    {1, 3, 2, 4, 0},
    {1, 3, 0, 4, 2},
    {1, 0, 3, 4, 2},
    {1, 0, 2, 4, 3},
    {2, 0, 1, 4, 3},
    {2, 0, 3, 4, 1},
    {2, 1, 0, 4, 3},
    {2, 1, 3, 4, 0},
    {2, 3, 0, 4, 1},
    {2, 3, 1, 4, 0},
    {3, 0, 1, 4, 2},
    {3, 0, 2, 4, 1},
    {3, 1, 0, 4, 2},
    {3, 1, 2, 4, 0},
    {3, 2, 0, 4, 1},
    {3, 2, 1, 4, 0},
    {0, 1, 4, 2, 3},
    {0, 1, 4, 3, 2},
    {0, 3, 4, 1, 2},
    {0, 3, 4, 2, 1},
    {0, 2, 4, 1, 3},
    {0, 2, 4, 3, 1},
    {1, 2, 4, 3, 0},
    {1, 2, 4, 0, 3},
    {1, 3, 4, 2, 0},
    {1, 3, 4, 0, 2},
    {1, 0, 4, 3, 2},
    {1, 0, 4, 2, 3},
    {2, 0, 4, 1, 3},
    {2, 0, 4, 3, 1},
    {2, 1, 4, 0, 3},
    {2, 1, 4, 3, 0},
    {2, 3, 4, 0, 1},
    {2, 3, 4, 1, 0},
    {3, 0, 4, 1, 2},
    {3, 0, 4, 2, 1},
    {3, 1, 4, 0, 2},
    {3, 1, 4, 2, 0},
    {3, 2, 4, 0, 1},
    {3, 2, 4, 1, 0},
    {0, 4, 1, 2, 3},
    {0, 4, 1, 3, 2},
    {0, 4, 3, 1, 2},
    {0, 4, 3, 2, 1},
    {0, 4, 2, 1, 3},
    {0, 4, 2, 3, 1},
    {1, 4, 2, 3, 0},
    {1, 4, 2, 0, 3},
    {1, 4, 3, 2, 0},
    {1, 4, 3, 0, 2},
    {1, 4, 0, 3, 2},
    {1, 4, 0, 2, 3},
    {2, 4, 0, 1, 3},
    {2, 4, 0, 3, 1},
    {2, 4, 1, 0, 3},
    {2, 4, 1, 3, 0},
    {2, 4, 3, 0, 1},
    {2, 4, 3, 1, 0},
    {3, 4, 0, 1, 2},
    {3, 4, 0, 2, 1},
    {3, 4, 1, 0, 2},
    {3, 4, 1, 2, 0},
    {3, 4, 2, 0, 1},
    {3, 4, 2, 1, 0},
    {4, 0, 1, 2, 3},
    {4, 0, 1, 3, 2},
    {4, 0, 3, 1, 2},
    {4, 0, 3, 2, 1},
    {4, 0, 2, 1, 3},
    {4, 0, 2, 3, 1},
    {4, 1, 2, 3, 0},
    {4, 1, 2, 0, 3},
    {4, 1, 3, 2, 0},
    {4, 1, 3, 0, 2},
    {4, 1, 0, 3, 2},
    {4, 1, 0, 2, 3},
    {4, 2, 0, 1, 3},
    {4, 2, 0, 3, 1},
    {4, 2, 1, 0, 3},
    {4, 2, 1, 3, 0},
    {4, 2, 3, 0, 1},
    {4, 2, 3, 1, 0},
    {4, 3, 0, 1, 2},
    {4, 3, 0, 2, 1},
    {4, 3, 1, 0, 2},
    {4, 3, 1, 2, 0},
    {4, 3, 2, 0, 1},
    {4, 3, 2, 1, 0}}
};

int fact (int x) {
    if (x == 0 || x == 1)
        return 1;
    return x*fact(x-1);
}

/*
 * The id is the maximum of all possible states with permutations of the input/output variables.
 * That way, we ensure that all functions that are identical up to input or output reordering have the same id and are treated only once.
 * Maximum is with the following order: most significant coefficient is in M[0][0], second most in M[0][1] etc.
 */
void compute_id (uint32_t ** id, algo_state * state) {
    int r_n, i_n;
    
    for (r_n=0; r_n<NB_REGISTERS; r_n++) {
        for (i_n=0; i_n<NB_INPUTS; i_n++) {
            id[r_n][i_n] = state->branch_vals[r_n][i_n];
        }
    }
    
    uint32_t M[NB_REGISTERS][NB_INPUTS];
    
    bool is_less, is_more;
    
    for (r_n=0; r_n<NB_REGISTERS; r_n++) {
        for (i_n=0; i_n<NB_INPUTS; i_n++) {
            M[r_n][i_n] = state->branch_vals[r_n][i_n];
        }
    }
    
    int input_order, output_order;
    for (input_order=0; input_order<fact(NB_INPUTS); input_order++) {
        for (output_order=0; output_order<fact(NB_REGISTERS); output_order++) {
            is_more = 0;
            is_less = 0;
            for (r_n=0; r_n<NB_REGISTERS && !(is_more | is_less); r_n++) {
                for (i_n=0; i_n<NB_INPUTS && !(is_more | is_less); i_n++) {
                    if (M[order_permutations[NB_REGISTERS][output_order][r_n]][order_permutations[NB_INPUTS][input_order][i_n]] == id[r_n][i_n]) {
                        continue;
                    }
                    if (M[order_permutations[NB_REGISTERS][output_order][r_n]][order_permutations[NB_INPUTS][input_order][i_n]] < id[r_n][i_n]) {
                        is_less = 1;
                        continue;
                    }
                    is_more = 1;                    
                }
            }
            if (is_more) { // Id of this reordered state is greater than the max, so it becomes the new max.
                for (r_n=0; r_n<NB_REGISTERS; r_n++) {
                    for (i_n=0; i_n<NB_INPUTS; i_n++) {
                        id[r_n][i_n] = M[order_permutations[NB_REGISTERS][output_order][r_n]][order_permutations[NB_INPUTS][input_order][i_n]];
                    }
                }
            }
        }
    }
}

bool test_MDS (uint32_t ** M, char selected_outputs[NB_INPUTS]) {
    /* Note that this function assumes that we have a N x 4 matrix M, and we select 4 lines, yielding a 4 x 4 matrix. */
    
    __m128i dim2det[NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS]; /* Not optimal in terms of memory, we could do with a [4][3][4][3] since the lines and columns must be different. */
    __m128i dim3det[NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS]; /* Not optimal in terms of memory. Again, lines and columns different, but also we could just store the eliminated lines and columns. */
    
    
    __m128i MM[NB_INPUTS][NB_INPUTS];
    
    int line1, line2, line3, column1, column2, column3;    
    
    /* Dimension 1 determinants != 0. */
    for (line1=0; line1<NB_INPUTS; line1++) {
        for (column1=0; column1<NB_INPUTS; column1++) {
            if (M[selected_outputs[line1]][column1] == 0) {
                return false;
            }
            MM[line1][column1] = _mm_set_epi32(0, 0, 0, M[selected_outputs[line1]][column1]);
        }
    }
    
    if (NB_INPUTS > 1) {
        /* Dimension 2 determinants != 0. */
        for (line1=0; line1<NB_INPUTS; line1++) {
            for (line2=line1+1; line2<NB_INPUTS; line2++) {
                for (column1=0; column1<NB_INPUTS; column1++) {
                    for (column2=column1+1; column2<NB_INPUTS; column2++) {
                        dim2det[line1][line2][column1][column2] = _mm_xor_si128(_mm_clmulepi64_si128(MM[line1][column1], MM[line2][column2], 0x00) , \
                                                                                        _mm_clmulepi64_si128(MM[line1][column2], MM[line2][column1], 0x00));
                        if (_mm_testz_si128(dim2det[line1][line2][column1][column2], dim2det[line1][line2][column1][column2])) // Test if det is zero.
                            return false;
                    }
                }
            }
        }
        
        if (NB_INPUTS > 2) {
            /* Dimension 3 determinants != 0. */
            for (line1=0; line1<NB_INPUTS; line1++) {
                for (line2=line1+1; line2<NB_INPUTS; line2++) {
                    for (line3=line2+1; line3<NB_INPUTS; line3++) {
                        for (column1=0; column1<NB_INPUTS; column1++) {
                            for (column2=column1+1; column2<NB_INPUTS; column2++) {
                                for (column3=column2+1; column3<NB_INPUTS; column3++) {
                                    dim3det[line1][line2][line3][column1][column2][column3] = _mm_xor_si128(
                                                                                                _mm_xor_si128(_mm_clmulepi64_si128(MM[line1][column1], dim2det[line2][line3][column2][column3], 0x00), \
                                                                                                                    _mm_clmulepi64_si128(MM[line1][column2], dim2det[line2][line3][column1][column3], 0x00)), \
                                                                                                _mm_clmulepi64_si128(MM[line1][column3], dim2det[line2][line3][column1][column2], 0x00));
                                    if (_mm_testz_si128(dim3det[line1][line2][line3][column1][column2][column3], dim3det[line1][line2][line3][column1][column2][column3])) // Test if det is zero.
                                        return false;
                                }
                            }
                        }
                    }
                }
            }
            
            if (NB_INPUTS > 3) {
                /* Dimension 4 determinant != 0. */
                /* Multiplying a 32-bit word with a 96-bit word takes work.
                * We use: Let u the 32-bit word, v||w the 96-bit word, with v on 32 bits, w on 64 bits.
                * r = (uxw) ^ [(uxv)<<64].
                */
                __m128i det4 = _mm_setzero_si128();
                __m64 zero64 = _mm_setzero_si64();
                __m128i v, w;
                v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[1][2][3][1][2][3], 1));
                w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[1][2][3][1][2][3], 0));
                __m128i mul1 = _mm_clmulepi64_si128(MM[0][0], w, 0x00);
                __m128i mul2 = _mm_clmulepi64_si128(MM[0][0], v, 0x00);
                mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
                det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
                
                v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[1][2][3][0][2][3], 1));
                w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[1][2][3][0][2][3], 0));
                mul1 = _mm_clmulepi64_si128(MM[0][1], w, 0x00);
                mul2 = _mm_clmulepi64_si128(MM[0][1], v, 0x00);
                mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
                det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
                
                v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[1][2][3][0][1][3], 1));
                w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[1][2][3][0][1][3], 0));
                mul1 = _mm_clmulepi64_si128(MM[0][2], w, 0x00);
                mul2 = _mm_clmulepi64_si128(MM[0][2], v, 0x00);
                mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
                det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
                
                v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[1][2][3][0][1][2], 1));
                w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[1][2][3][0][1][2], 0));
                mul1 = _mm_clmulepi64_si128(MM[0][3], w, 0x00);
                mul2 = _mm_clmulepi64_si128(MM[0][3], v, 0x00);
                mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
                det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
                
                if (_mm_testz_si128(det4, det4))
                    return false;
            }
        }
    }
    return true;
}

void clean_state (algo_state * state) {
    for (int i=0; i<NB_REGISTERS; i++)
        free(state->branch_vals[i]);
    free(state->branch_vals);
    state->branch_vals = NULL;
}

void free_state (algo_state * state) {
    for (int i=0; i<NB_REGISTERS; i++)
        free(state->branch_vals[i]);
    free(state->branch_vals);
    free(state->op);
    free(state);
}

void compute_state (algo_state * current_state) {
    int branch_n, input_n;
    // Parent state.
    current_state->branch_vals = (uint32_t**) malloc (NB_REGISTERS*sizeof(uint32_t*));
    for (branch_n=0; branch_n<NB_REGISTERS; branch_n++) {
        current_state->branch_vals[branch_n] = (uint32_t*) malloc (NB_INPUTS*sizeof(uint32_t));
        for (input_n=0; input_n<NB_INPUTS; input_n++) {
            current_state->branch_vals[branch_n][input_n] = current_state->pred->branch_vals[branch_n][input_n];
        }
    }
    // Plus operation.
    if (current_state->op->type==XOR) {
        if (current_state->op->from < NB_REGISTERS) {
            for (input_n=0; input_n<NB_INPUTS; input_n++) {
                current_state->branch_vals[current_state->op->to][input_n] ^= current_state->branch_vals[current_state->op->from][input_n];
            }
        }
        else {
            current_state->branch_vals[current_state->op->to][current_state->op->from-NB_REGISTERS] ^= 1;
        }
    }
    else if (current_state->op->type==MUL) {
        for (input_n=0; input_n<NB_INPUTS; input_n++) {
            bool not_zero = true;
            if (current_state->branch_vals[current_state->op->from][input_n] == 0)
                not_zero = false;
            current_state->branch_vals[current_state->op->from][input_n] <<= 1;
            if (current_state->branch_vals[current_state->op->from][input_n] == 0 && not_zero) { // Overflow.
                printf ("Overflow !! Exiting.\n");
                //exit(1);
            }
        }
    }
    else {
        if (current_state->op->from < NB_REGISTERS) {
            for (input_n=0; input_n<NB_INPUTS; input_n++) {
                current_state->branch_vals[current_state->op->to][input_n] = current_state->branch_vals[current_state->op->from][input_n];
            }
        }
        else {
            for (input_n=0; input_n<NB_INPUTS; input_n++) {
                current_state->branch_vals[current_state->op->to][input_n] = 0;
            }
            current_state->branch_vals[current_state->op->to][current_state->op->from-NB_REGISTERS] = 1;
        }
    }
}

void test_restrictions_MDS (algo_state * current_state) {
    char selected_outputs[NB_INPUTS], i;
    algo_state * pred;
    
    for (selected_outputs[0]=0; selected_outputs[0]<NB_REGISTERS; selected_outputs[0]++) {
        for (selected_outputs[1]=selected_outputs[0]+1; selected_outputs[1]<NB_REGISTERS; selected_outputs[1]++) {
            for (selected_outputs[2]=selected_outputs[1]+1; selected_outputs[2]<NB_REGISTERS; selected_outputs[2]++) {
#if NB_INPUTS>3
                for (selected_outputs[3]=selected_outputs[2]+1; selected_outputs[3]<NB_REGISTERS; selected_outputs[3]++) { // All possible choices of 4 output branches.
#endif
                    if (test_MDS(current_state->branch_vals, selected_outputs)) {
                        printf ("Found MDS !!!\n");
                        print_state(current_state, 1, 1, true, true);
                        printf ("Weight is %d\n", (int)current_state->weight);
                        printf ("Selected outputs :\n\t( ");
                        for (i=0; i<NB_INPUTS; i++)
                            printf ("%d ", selected_outputs[i]);
                        printf (")\n");
                        exit(0);
                    }
#if NB_INPUTS>3
                }
#endif
            }
        }
    }
}

/*
 * For every columns having a 0, we will need at least 1 XOR to have an MDS matrix.
 */
char min_dist_to_MDS (uint32_t ** M) {
    char weight = CHAR_MAX;
    int nb_columns_having_zero = 0;
    for (int i=0; i<NB_REGISTERS; i++) {
        for (int j=0; j<NB_INPUTS; j++) {
            if (M[i][j] == 0) {
                nb_columns_having_zero++;
                break;
            }
        }
    }
    return (nb_columns_having_zero==0?0:(nb_columns_having_zero-(NB_REGISTERS-NB_INPUTS)) * XOR_WEIGHT);
}

void spawn_next_states (algo_state * current_state, std::priority_queue<algo_state*, std::vector<algo_state*>, QueueComparator> * remaining_states, std::unordered_set<uint32_t**, AlgoHasher, HashComparator> * scanned_states) {
    int type_of_op_int, to, from, i, j;
    enum op_name type_of_op;
    
    uint32_t ** id = (uint32_t**) malloc (NB_REGISTERS*sizeof(uint32_t*));
    for (i=0; i<NB_REGISTERS; i++)
        id[i] = (uint32_t*) malloc (NB_INPUTS*sizeof(uint32_t));
    
    char selected_outputs[NB_INPUTS];
    
    for (type_of_op_int=XOR; type_of_op_int<=CPY; type_of_op_int++) {
        // No 2 copies in a row.
        type_of_op = (enum op_name) type_of_op_int;
        if (current_state->op != NULL && current_state->op->type == CPY && type_of_op == CPY) {
            continue;
        }
        for (to=0; to<NB_REGISTERS; to++) {
            // After a copy, the copy must be the destination of the operation.
            if (current_state->op != NULL && current_state->op->type == CPY && current_state->op->to != to)
                continue;
            for (from=(type_of_op==MUL?to:0); from<(type_of_op==MUL?to+1:NB_REGISTERS+(KEEP_INPUTS?NB_INPUTS:0)); from++) {
                if (type_of_op!=MUL && to==from)
                    continue;
                algo_op * new_op = (algo_op*)malloc(sizeof(algo_op));
                new_op->type = type_of_op;
                new_op->from = from;
                new_op->to = to;
                algo_state * next_state = (algo_state*)malloc(sizeof(algo_state));
                next_state->pred = current_state;
                next_state->op = new_op;
                next_state->weight = current_state->weight + (type_of_op==XOR?XOR_WEIGHT:(type_of_op==MUL?MUL_WEIGHT:CPY_WEIGHT));
                next_state->branch_vals = NULL;
                
                /* We filter here 2 things:
                 *  - injectivity (can only change after a copy).
                 *  - equivalence up to input/output reordering (id).
                 * Note that the id test at this point can only test if we have already SCANNED a state with the same id as next_state.
                 * We cannot test if another state with the same id is in the queue since we don't store the id in the queue (to save memory).
                 */
                compute_state(next_state);
                if (type_of_op == CPY) {
                    // Injectivity test assumes that NB_REGISTERS = NB_INPUTS+1;
                    for (i=0, j=0; i<NB_REGISTERS; i++)
                        if (i != to) {
                            selected_outputs[j] = i;
                            j++;
                        }
                    if (!test_injective(next_state->branch_vals, selected_outputs)) {
                        free_state(next_state);
                        continue;
                    }
                }
#ifdef COMPUTE_ID_FIRST
                compute_id(id, next_state);
                if (scanned_states->find(id) != scanned_states->end()) { // Id already scanned.
                    free_state(next_state);
                    continue;
                }
#endif                
                // Computing a bound on the distance to an MDS matrix.
                next_state->weight_to_MDS = min_dist_to_MDS(current_state->branch_vals);
                
                // next_state->branch_vals will be STORED later, when next_state will be treated (as the new current_state). That way, we don't need to store the branch_vals for all this node's sons (77 sons).
                clean_state(next_state);
                
                (*remaining_states).push(next_state);
                /*
                if (current_state->op != NULL)
                    printf ("Current state op : %c (%d, %d)\n", (current_state->op->type==XOR?'x':(current_state->op->type==MUL?'m':'c')), current_state->op->from, current_state->op->to);
                else printf ("Current state : no op\n");
                printf ("New state op : %c (%d, %d)\n", (next_state->op->type==XOR?'x':(next_state->op->type==MUL?'m':'c')), next_state->op->from, next_state->op->to);
                if (current_state->pred != NULL && current_state->pred->op != NULL)
                    printf ("Pred state op : %c (%d, %d)\n", (current_state->pred->op->type==XOR?'x':(current_state->pred->op->type==MUL?'m':'c')), current_state->pred->op->from, current_state->pred->op->to);
                else printf ("Pred : origin\n");*/
            }
        }
    }
    
    for (i=0; i<NB_REGISTERS; i++)
        free(id[i]);
    free(id);
}

int main () {
    std::priority_queue<algo_state*, std::vector<algo_state*>, QueueComparator> remaining_states;
    std::unordered_set<uint32_t**, AlgoHasher, HashComparator> scanned_states;
    
    algo_state * current_state = (algo_state*)malloc(sizeof(algo_state));
    
    int i, j;
    
    current_state->branch_vals = (uint32_t**) malloc(NB_REGISTERS*sizeof(uint32_t*));
    for (i=0; i<NB_REGISTERS; i++) {
        current_state->branch_vals[i] = (uint32_t*) calloc (NB_INPUTS, sizeof(uint32_t));
        if (i<NB_INPUTS)
             current_state->branch_vals[i][i] = 1; // Comment to start with a null state.
    }
    current_state->weight = 0;
    current_state->weight_to_MDS = NB_INPUTS*XOR_WEIGHT;
    current_state->pred = NULL;
    current_state->op = NULL;
    remaining_states.push(current_state);
    
    uint32_t nb_scanned = 0, nb_tested = 0;
    int current_weight = 0;
    
    uint32_t ** id = (uint32_t**) malloc (NB_REGISTERS*sizeof(uint32_t*));
    for (i=0; i<NB_REGISTERS; i++)
        id[i] = (uint32_t*) malloc (NB_INPUTS*sizeof(uint32_t));
    
    scanned_states.max_load_factor(1);
    
    while (true) {
        nb_tested++;
        current_state = remaining_states.top(); // Next function to test.
        remaining_states.pop();
        if (current_state->branch_vals == NULL) { // Computing the current state value from the father's state.
            compute_state(current_state);
        }
        if (current_state->weight + current_state->weight_to_MDS > current_weight) { // Printing when we get to a new weight.
            printf ("New weight : %d (%d, +%d to MDS)\n", current_state->weight + current_state->weight_to_MDS, current_state->weight, current_state->weight_to_MDS);
            printf ("Number of distinct ids : %" PRIu32 "/%" PRIu32 "(1/%.1f)\n", nb_scanned, nb_tested, (nb_scanned?(float)nb_tested/nb_scanned:0));
            print_state(current_state, nb_scanned, nb_tested, false, true);
            current_weight = current_state->weight + current_state->weight_to_MDS;
            printf ("Scanned size : %lu\n", scanned_states.size());
            printf ("Remaining size : %lu\n", remaining_states.size());
        }
        compute_id(id, current_state); // Get a unique id invariant under input/output reordering.
        if (scanned_states.find(id) == scanned_states.end()) { // Current state not scanned yet (even up to input/output reordering).
            test_restrictions_MDS(current_state); // Test if any restriction to 4 output branches is MDS. If so, prints and ends.
            // Checking id.
            uint32_t ** id_c = (uint32_t**) malloc (NB_REGISTERS*sizeof(uint32_t*));
            for (i=0; i<NB_REGISTERS; i++) {
                id_c[i] = (uint32_t*) malloc (NB_INPUTS*sizeof(uint32_t));
                for (j=0; j<NB_INPUTS; j++)
                    id_c[i][j] = id[i][j];
            }
            scanned_states.insert(id_c);
            nb_scanned++;
            // Computing all children states.
            spawn_next_states(current_state, &remaining_states, &scanned_states);
        }
        else {
            // Id was already checked, so an equivalent state was already analysed. Freeing.
            free_state(current_state);
        }
    }
    
    //exit(0);
}
