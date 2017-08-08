/* NOTE:
   This codes makes some assumptions:
   - NB_REGISTERS = NB_INPUTS + 1 (can be removed?)
   - NB_INPUTS is 3 or 4
   - NB_REGISTERS is less than 8
*/

#define __STDC_FORMAT_MACROS 1
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <numeric>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_set.h>
#include <boost/functional/hash.hpp>
#include "wmmintrin.h"
#include <smmintrin.h>
#include <algorithm>

#define stringify(x) stringify_(x)
#define stringify_(x) #x
#define CATCH                                                           \
    catch (const std::exception& ex) {                                  \
        fprintf(stderr, "Error occured line " stringify(__LINE__)       \
                ": %s\n", ex.what());                                   \
        sleep(1);                                                       \
        exit(-1);                                                       \
    } catch(...) {                                                      \
        fprintf(stderr, "Unknown error occured line "                   \
                stringify(__LINE__) "\n");                              \
        sleep(1);                                                       \
        exit(-1);                                                       \
    }

#define NB_INPUTS 4
#define NB_REGISTERS 5
#define KEEP_INPUTS 0

#define XOR_WEIGHT 2
#define MUL_WEIGHT 1
#define CPY_WEIGHT 0

#define MAX_WEIGHT (1 + 8*XOR_WEIGHT + 3*MUL_WEIGHT)

#define COMPUTE_ID_FIRST
//#define REMEMBER_MATRIX

// uint32_t is considering we won't have more than about 20 multiplications, therefore we need at least 20 bits not to overflow.
typedef std::array<std::array<uint32_t, NB_INPUTS>, NB_REGISTERS> matrix;
typedef tbb::concurrent_unordered_set<matrix, boost::hash<matrix>> matrix_set;

matrix identity() {
    matrix m;
    for (int i=0; i<NB_REGISTERS; i++) {
        for (int j=0; j<NB_INPUTS; j++) {
            m[i][j] = i==j? 1: 0;
        }
    }
    return m;
}

void print_matrix(matrix m) {
    for (const auto &l : m) {
        for (const auto &x : l) {
            printf ("%i ", x);
        }
        printf ("\n");
    }
}

enum op_name {XOR=0, MUL=1, CPY=2, NONE=3};

typedef struct {
    enum op_name type;
    char from, to;
} algo_op;

class AlgoState {
    AlgoState * pred;
    algo_op op;
public:
    char weight, weight_to_MDS;

    AlgoState(): AlgoState(NULL, (algo_op){NONE, 0, 0}, 0, NB_INPUTS*XOR_WEIGHT) {};
    AlgoState(AlgoState *_pred, algo_op _op, char _weight, char _weight_to_MDS): pred(_pred), op(_op), weight(_weight), weight_to_MDS(_weight_to_MDS) {};

    virtual matrix branch_vals();
    void print_state (int nb_scanned, int nb_tested, int print_all, int print_pred);
    bool test_restrictions_MDS ();
    void spawn_next_states (tbb::concurrent_queue<AlgoState>* remaining_states, matrix_set& scanned_states);
    int queue_weight() const { return weight + weight_to_MDS; }
    bool operator <(const AlgoState &other) const {
        return queue_weight() > other.queue_weight();
    }
};

class AlgoStateMatrix : public AlgoState {
    matrix bv;
public:
    AlgoStateMatrix(AlgoState s): AlgoState(s) {
        bv = AlgoState::branch_vals();
    }
    // Default contructor: intial state
    AlgoStateMatrix(): AlgoState() {
        bv = identity();
    }
    matrix branch_vals() { return bv; }
};

typedef tbb::concurrent_queue<AlgoState> state_queue;

void AlgoState::print_state (int nb_scanned, int nb_tested, int print_all, int print_pred) {
    int depth = 0;
    AlgoState * depth_finder;
    for (depth_finder=this; depth_finder->pred != NULL; depth_finder=depth_finder->pred, depth++);
    if (print_all) {
        printf ("Number of distinct ids (weight=%3d, depth=%3d) : %" PRIu32 "/%" PRIu32 "(1/%.1f)\n", weight, depth, nb_scanned, nb_tested, (nb_scanned?(float)nb_tested/nb_scanned:0));
    }
        
    if (print_all) {
        int i, input_n;
        matrix bv = branch_vals();
        printf ("Current state\n");
        for (i=0; i<NB_REGISTERS; i++) {
            for (input_n=0; input_n<NB_INPUTS; input_n++)
                printf ("%d ", bv[i][input_n]);
            printf ("\n");
        }
        if (op.type != NONE)
            printf ("Current state op : %c (%d, %d)\n", (op.type==XOR?'x':(op.type==MUL?'m':'c')), op.from, op.to);
    }
    if (print_pred) {
        printf ("Printing preds\n#############################\n");
        for (depth_finder=this; depth_finder->pred != NULL; depth_finder=depth_finder->pred) {
            printf ("%c (%d, %d)\n", (depth_finder->op.type==XOR?'x':(depth_finder->op.type==MUL?'m':'c')), depth_finder->op.from, depth_finder->op.to);
            if (print_all)
                depth_finder->pred->print_state(nb_scanned, nb_tested, false, false);
        }
    }
    printf ("End of state\n");
}

bool test_injective (matrix M, char selected_outputs[NB_INPUTS]) {
    /* Note that this function assumes that we have a N x 4 matrix M, and we select 4 lines, yielding a 4 x 4 matrix. 
     * This test only happens after adding a copy operation (only operation that can change injectivity).
     * In that case, the matrix gets 2 identical lines.
     * We consider that NB_REGISTERS = NB_INPUTS + 1, therefore we are left with a square matrix.
     * We simply compute its determinant and return if it is 0 or not.
     */
    
    __m128i dim2det[NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS]; /* Not optimal in terms of memory, we could do with a [4][3][4][3] since the lines and columns must be different. */
    __m128i dim3det[NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS][NB_INPUTS]; /* Not optimal in terms of memory. Again, lines and columns different, but also we could just store the eliminated lines and columns. */
    
    
    __m128i MM[NB_INPUTS][NB_INPUTS];
    
    int line1, column1, column2, column3;
    
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

int order_permutations[6][120][5] = {
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

#define _mm_is_zero(x) _mm_testz_si128(x,x)

/*
 * The id is the maximum of all possible states with permutations of the input/output variables.
 * That way, we ensure that all functions that are identical up to input or output reordering have the same id and are treated only once.
 * Maximum is with the following order: most significant coefficient is in M[0][0], second most in M[0][1] etc.
 */
matrix compute_id (matrix id) {
    matrix max = id;

    // Look for columns that contain max coefficient
    int max_cols[NB_INPUTS] = {0};
    uint32_t max_coeff = 0;
    for (int i=0; i<NB_REGISTERS; i++) {
        for (int j=0; j<NB_INPUTS; j++) {
            if (id[i][j] > max_coeff) {
                memset(max_cols, 0, sizeof(max_cols));
                max_coeff = id[i][j];
            }
            if (id[i][j] == max_coeff) {
                max_cols[j] = 1;
            }
        }
    }
    for (int order=0; order<fact(NB_INPUTS); order++) {
        if (max_cols[order_permutations[NB_INPUTS][order][0]] == 0)
            continue; // Max coefficient not in forst column
        matrix tmp;
        for (int i=0; i<NB_REGISTERS; i++) {
            for (int j=0; j<NB_INPUTS; j++) {
                tmp[i][j] = id[i][order_permutations[NB_INPUTS][order][j]];
            }
        }
        std::sort(tmp.rbegin(), tmp.rend());
        if (tmp > max)
            max = tmp;
    }

    return max;
}

// With zero == true:  returns smallest zero minor OF BEST SQUARE MATRIX
// With zero == false: returns largest non-zero minor
int test_minors (bool zero, matrix M) {
    /* Note that this function assumes that we have a N x 4 matrix M, and we select 4 lines, yielding a 4 x 4 matrix. */
    
    __m128i dim2det[NB_REGISTERS][NB_REGISTERS][NB_INPUTS][NB_INPUTS];
    __m128i dim3det[NB_REGISTERS][NB_REGISTERS][NB_REGISTERS][NB_INPUTS][NB_INPUTS][NB_INPUTS];
    
    __m128i MM[NB_REGISTERS][NB_INPUTS];

    int line1, line2, line3, column1, column2, column3;    
    int max = 0;

    std::array<bool, NB_REGISTERS> maybeMDSwithout;
    for (int i=0; i<NB_REGISTERS; i++)
        maybeMDSwithout[i] = true;
    
    /* Dimension 1 determinants != 0. */
    for (line1=0; line1<NB_REGISTERS; line1++) {
        for (column1=0; column1<NB_INPUTS; column1++) {
            MM[line1][column1] = _mm_set_epi32(0, 0, 0, M[line1][column1]);
            if (!zero && M[line1][column1])
                max = std::max(max, 1);
            if (zero && !M[line1][column1]) {
                for (int skip = 0; skip<NB_REGISTERS; skip++) {
                    if (skip != line1)
                        maybeMDSwithout[skip] = false;
                }
	    }
        }
    }
    if (zero && !std::accumulate(maybeMDSwithout.begin(), maybeMDSwithout.end(), 0))
        return 1;
    
    if (NB_INPUTS > 1) {
        /* Dimension 2 determinants != 0. */
        for (line1=0; line1<NB_REGISTERS; line1++) {
            for (line2=line1+1; line2<NB_REGISTERS; line2++) {
                for (column1=0; column1<NB_INPUTS; column1++) {
                    for (column2=column1+1; column2<NB_INPUTS; column2++) {
                        dim2det[line1][line2][column1][column2] = _mm_xor_si128(_mm_clmulepi64_si128(MM[line1][column1], MM[line2][column2], 0x00) , \
                                                                                _mm_clmulepi64_si128(MM[line1][column2], MM[line2][column1], 0x00));
                        if (!zero && !_mm_testz_si128(dim2det[line1][line2][column1][column2], dim2det[line1][line2][column1][column2])) // Test if det is zero.
                            max = std::max(max, 2);
                        if (zero && _mm_testz_si128(dim2det[line1][line2][column1][column2], dim2det[line1][line2][column1][column2])) {
                            for (int skip = 0; skip<NB_REGISTERS; skip++) {
                                if (skip != line1 && skip != line2)
                                    maybeMDSwithout[skip] = false;
                            }
			}
                    }
                }
            }
        }
        if (zero && !std::accumulate(maybeMDSwithout.begin(), maybeMDSwithout.end(), 0))
            return 2;
    }
        
    if (NB_INPUTS > 2) {
        /* Dimension 3 determinants != 0. */
        for (line1=0; line1<NB_REGISTERS; line1++) {
            for (line2=line1+1; line2<NB_REGISTERS; line2++) {
                for (line3=line2+1; line3<NB_REGISTERS; line3++) {
                    for (column1=0; column1<NB_INPUTS; column1++) {
                        for (column2=column1+1; column2<NB_INPUTS; column2++) {
                            for (column3=column2+1; column3<NB_INPUTS; column3++) {
                                dim3det[line1][line2][line3][column1][column2][column3] = _mm_xor_si128(
                                                                                                        _mm_xor_si128(_mm_clmulepi64_si128(MM[line1][column1], dim2det[line2][line3][column2][column3], 0x00), \
                                                                                                                      _mm_clmulepi64_si128(MM[line1][column2], dim2det[line2][line3][column1][column3], 0x00)), \
                                                                                                        _mm_clmulepi64_si128(MM[line1][column3], dim2det[line2][line3][column1][column2], 0x00));
                                if (!zero && !_mm_testz_si128(dim3det[line1][line2][line3][column1][column2][column3], dim3det[line1][line2][line3][column1][column2][column3])) // Test if det is zero.
                                    max = std::max(max, 3);
                                if (zero && _mm_testz_si128(dim3det[line1][line2][line3][column1][column2][column3], dim3det[line1][line2][line3][column1][column2][column3])) {
                                    for (int skip = 0; skip<NB_REGISTERS; skip++) {
                                        if (skip != line1 && skip != line2 && skip != line3)
                                            maybeMDSwithout[skip] = false;
                                    }
				}
                            }
                        }
                    }
                }
            }
        }

        if (zero && !std::accumulate(maybeMDSwithout.begin(), maybeMDSwithout.end(), 0))
            return 3;
    }
    if (NB_INPUTS > 3) {
        /* Dimension 4 determinant != 0. */
        for (int skip = 0; skip<NB_REGISTERS; skip++) {
            int lines[4];
            for (int i=0; i<4; i++)
                lines[i] = i<skip? i: i+1;

            /* Multiplying a 32-bit word with a 96-bit word takes work.
             * We use: Let u the 32-bit word, v||w the 96-bit word, with v on 32 bits, w on 64 bits.
             * r = (uxw) ^ [(uxv)<<64].
             */
            __m128i det4 = _mm_setzero_si128();
            __m64 zero64 = _mm_setzero_si64();
            __m128i v, w;
            v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[lines[1]][lines[2]][lines[3]][1][2][3], 1));
            w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[lines[1]][lines[2]][lines[3]][1][2][3], 0));
            __m128i mul1 = _mm_clmulepi64_si128(MM[lines[0]][0], w, 0x00);
            __m128i mul2 = _mm_clmulepi64_si128(MM[lines[0]][0], v, 0x00);
            mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
            det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
            
            v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[lines[1]][lines[2]][lines[3]][0][2][3], 1));
            w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[lines[1]][lines[2]][lines[3]][0][2][3], 0));
            mul1 = _mm_clmulepi64_si128(MM[lines[0]][1], w, 0x00);
            mul2 = _mm_clmulepi64_si128(MM[lines[0]][1], v, 0x00);
            mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
            det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
            
            v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[lines[1]][lines[2]][lines[3]][0][1][3], 1));
            w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[lines[1]][lines[2]][lines[3]][0][1][3], 0));
            mul1 = _mm_clmulepi64_si128(MM[lines[0]][2], w, 0x00);
            mul2 = _mm_clmulepi64_si128(MM[lines[0]][2], v, 0x00);
            mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
            det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));
            
            v = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[lines[1]][lines[2]][lines[3]][0][1][2], 1));
            w = _mm_set_epi64(zero64, (__m64)_mm_extract_epi64(dim3det[lines[1]][lines[2]][lines[3]][0][1][2], 0));
            mul1 = _mm_clmulepi64_si128(MM[lines[0]][3], w, 0x00);
            mul2 = _mm_clmulepi64_si128(MM[lines[0]][3], v, 0x00);
            mul2 = _mm_slli_si128(mul2, 8); // Shift left 64 bits.
            det4 = _mm_xor_si128(det4, _mm_xor_si128(mul1, mul2));

            if (!zero && !_mm_testz_si128(det4, det4))
                max = std::max(max, 4);
            if (zero && _mm_testz_si128(det4, det4))
                maybeMDSwithout[skip] = false;
        }

        if (zero && !std::accumulate(maybeMDSwithout.begin(), maybeMDSwithout.end(), 0))
            return 4;
    }

    if (zero)
        return NB_INPUTS+1;
    else
        return max;
}

// rank is largest non-zero minor
int rank (matrix M) {
    return test_minors(false, M);
}


matrix AlgoState::branch_vals() {
    if (op.type == NONE)
        return identity();
    
    int input_n;
    // Parent state.
    matrix bv = pred->branch_vals();
    // Plus operation.
    if (op.type==XOR) {
        if (op.from < NB_REGISTERS) {
            for (input_n=0; input_n<NB_INPUTS; input_n++) {
                bv[op.to][input_n] ^= bv[op.from][input_n];
            }
        }
        else {
            bv[op.to][op.from-NB_REGISTERS] ^= 1;
        }
    }
    else if (op.type==MUL) {
        for (input_n=0; input_n<NB_INPUTS; input_n++) {
            bool not_zero = true;
            if (bv[op.from][input_n] == 0)
                not_zero = false;
            bv[op.from][input_n] <<= 1;
            if (bv[op.from][input_n] == 0 && not_zero) { // Overflow.
                printf ("Overflow !! Exiting.\n");
                //exit(1);
            }
        }
    }
    else {
        if (op.from < NB_REGISTERS) {
            for (input_n=0; input_n<NB_INPUTS; input_n++) {
                bv[op.to][input_n] = bv[op.from][input_n];
            }
        }
        else {
            for (input_n=0; input_n<NB_INPUTS; input_n++) {
                bv[op.to][input_n] = 0;
            }
            bv[op.to][op.from-NB_REGISTERS] = 1;
        }
    }
    return bv;
}


bool AlgoState::test_restrictions_MDS () {
    matrix bv = branch_vals();

    if (test_minors(true, bv) == NB_INPUTS+1) {
#pragma omp critical
        {
            printf ("Found MDS !!!\n");
            print_state(1, 1, true, true);
            printf ("Weight is %d\n", (int)weight);
            fflush(stdout);
        }
        return true;
    }
    return false;
}

/*
 * For every columns having a 0, we will need at least 1 XOR to have an MDS matrix.
 * More generally, we consider the rank of columns without zeroes
 */
char min_dist_to_MDS (matrix M) {
    // Clear columns that contain a zero
    for (int i=0; i<NB_REGISTERS; i++) {
        bool has_zero = false;
        for (int j=0; j<NB_INPUTS; j++) {
            if (M[i][j] == 0) {
                has_zero = true;
                break;
            }
        }
        if (has_zero) {
            for (int j=0; j<NB_INPUTS; j++)
                M[i][j] = 0;
        }
    }

    return XOR_WEIGHT * (NB_INPUTS-rank(M));
}

void AlgoState::spawn_next_states (state_queue* remaining_states, matrix_set& scanned_states) {
    int type_of_op_int, to, from, i, j;
    enum op_name type_of_op;
    
    char selected_outputs[NB_INPUTS];
    
    for (type_of_op_int=XOR; type_of_op_int<=CPY; type_of_op_int++) {
        // No 2 copies in a row.
        type_of_op = (enum op_name) type_of_op_int;
        if (op.type == CPY && type_of_op == CPY) {
            continue;
        }
        for (to=0; to<NB_REGISTERS; to++) {
            // After a copy, the copy must be the destination of the operation.
            if (op.type == CPY && op.to != to)
                continue;
            for (from=(type_of_op==MUL?to:0); from<(type_of_op==MUL?to+1:NB_REGISTERS+(KEEP_INPUTS?NB_INPUTS:0)); from++) {
                if (type_of_op!=MUL && to==from)
                    continue;
                int new_weight = weight + (type_of_op==XOR?XOR_WEIGHT:(type_of_op==MUL?MUL_WEIGHT:CPY_WEIGHT));
                if (new_weight >= MAX_WEIGHT)
                    continue;

                AlgoState next_state(this, (algo_op){type_of_op, (char)from, (char)to}, new_weight, 0);
                
                /* We filter here 2 things:
                 *  - injectivity (can only change after a copy).
                 *  - equivalence up to input/output reordering (id).
                 * Note that the id test at this point can only test if we have already SCANNED a state with the same id as next_state.
                 * We cannot test if another state with the same id is in the queue since we don't store the id in the queue (to save memory).
                 */
                matrix bv = next_state.branch_vals();
                if (type_of_op == CPY) {
                    // Injectivity test assumes that NB_REGISTERS = NB_INPUTS+1;
                    for (i=0, j=0; i<NB_REGISTERS; i++)
                        if (i != to) {
                            selected_outputs[j] = i;
                            j++;
                        }
                    if (!test_injective(bv, selected_outputs)) {
                        continue;
                    }
                }
#ifdef COMPUTE_ID_FIRST
                matrix id = compute_id(bv);
                if (scanned_states.find(id) != scanned_states.end()) { // Id already scanned.
                    continue;
                }
#endif                
                // Computing a bound on the distance to an MDS matrix.
                next_state.weight_to_MDS = min_dist_to_MDS(bv);
                
                // next_state.branch_vals will be STORED later, when next_state will be treated (as the new current_state). That way, we don't need to store the branch_vals for all this node's sons (77 sons).
                assert(next_state.queue_weight() >= queue_weight());
                if (next_state.queue_weight() < MAX_WEIGHT) {
                    try {
                        remaining_states[next_state.queue_weight()].push(next_state);
                    } CATCH;
                }
                /*
                  if (op != NULL)
                  printf ("Current state op : %c (%d, %d)\n", (op.type==XOR?'x':(op.type==MUL?'m':'c')), op.from, op.to);
                  else printf ("Current state : no op\n");
                  printf ("New state op : %c (%d, %d)\n", (next_state.op.type==XOR?'x':(next_state.op.type==MUL?'m':'c')), next_state.op.from, next_state.op.to);
                  if (pred != NULL && pred->op != NULL)
                  printf ("Pred state op : %c (%d, %d)\n", (pred->op.type==XOR?'x':(pred->op.type==MUL?'m':'c')), pred->op.from, pred->op.to);
                  else printf ("Pred : origin\n");*/
            }
        }
    }
}

int main () {
    state_queue remaining_states[MAX_WEIGHT];
    matrix_set scanned_ids;

    AlgoState initial_state = AlgoStateMatrix();
    remaining_states[0].push(initial_state);
    
    int nb_scanned = 0, nb_tested = 0;
    //    scanned_states.max_load_factor(1);

    try {
        for (int current_weight=0; current_weight<MAX_WEIGHT; current_weight++) {
            printf ("New weight : %d\n", current_weight);
            printf ("Number of distinct ids : %" PRIu32 "/%" PRIu32 "(1/%.1f)\n", nb_scanned, nb_tested, (nb_scanned?(float)nb_tested/nb_scanned:0));
            printf ("Scanned size : %lu\n", scanned_ids.size());
            //                printf ("Remaining size : %lu\n", remaining_states[w].size());
            fflush(stdout);
        
            int old_nb_scanned = nb_scanned;
            nb_scanned = 0;
            int old_nb_tested = nb_tested;
            nb_tested = 0;
        
#pragma omp parallel reduction(+:nb_scanned) reduction(+:nb_tested)
            {
                // Next function to test.
                AlgoState pop_state;

                while (remaining_states[current_weight].try_pop(pop_state)) {
                    AlgoStateMatrix current_state(pop_state);
                    nb_tested++;
                    matrix id = compute_id(current_state.branch_vals()); // Get a unique id invariant under input/output reordering.
                    bool is_new;
                    try {
                        is_new = scanned_ids.insert(id).second;
                    } CATCH;
                    if (is_new) { // Current state not scanned yet (even up to input/output reordering).
                        nb_scanned++;
                        if (current_state.test_restrictions_MDS()) // Test if any restriction to 4 output branches is MDS. If so, prints and ends.
                            continue;
                        // Checking id.
                        // Insert into scanned_states
                        AlgoState *tmp;
                        try {
#ifdef REMEMBER_MATRIX
                            tmp = new AlgoStateMatrix(current_state);
#else
                            tmp = new AlgoState(current_state);
#endif
                        } CATCH;
                        // Computing all children states.
                        tmp->spawn_next_states(remaining_states, scanned_ids);
                    }
                    else {
                        // Id was already checked, so an equivalent state was already analysed. Do nothing.
                    }
                }
            }
            nb_scanned += old_nb_scanned;
            nb_tested += old_nb_tested;
        }
    } CATCH;
    
    printf("Reached MAX_WEIGHT\n");
    //exit(0);
}

/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-basic-offset: 4 */
/* End: */
