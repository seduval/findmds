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
#include <tbb/concurrent_vector.h>
#include <boost/functional/hash.hpp>
#include "wmmintrin.h"
#include <smmintrin.h>
#include <algorithm>

#include <sys/time.h>
#include <sys/resource.h>

/***********************************************************************
 *                         CONFIGURATION                               *
 ***********************************************************************/

#define NB_INPUTS 4
#define NB_REGISTERS 5

#define XOR_WEIGHT 2
#define MUL_WEIGHT 1
#define CPY_WEIGHT 0

// Note: MAX is excluded
#define MAX_WEIGHT (1 + 8*XOR_WEIGHT + 5*MUL_WEIGHT)
#define MAX_DEPTH 4

// Optimize depth first, rather than weight
// #define DEPTH_FIRST

// Uncomment to activate options
// #define KEEP_INPUTS
// #define TRY_DIV
// #define INDEP_MUL
 #define DIFFERENT_MUL

// You should leave this on
#define COMPUTE_ID_FIRST
#define PRINT_ALL_MDS

#define SLEEP_TIME 30

/***********************************************************************
 *                       END CONFIGURATION                             *
 ***********************************************************************/

#define MAX_QUEUE_WEIGHT (MAX_WEIGHT*MAX_DEPTH)


#ifdef TRY_DIV
#define INIT_VAL 0x10000
#else
#define INIT_VAL 1
#endif

// uint32_t is considering we won't have more than about 20 multiplications, therefore we need at least 20 bits not to overflow.
typedef std::array<std::array<uint32_t, NB_INPUTS>, NB_REGISTERS> matrix;
typedef tbb::concurrent_unordered_set<matrix, boost::hash<matrix>> matrix_set;

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

matrix init_matrix() {
    matrix m;
    for (int i=0; i<NB_REGISTERS; i++) {
        for (int j=0; j<NB_INPUTS; j++) {
            m[i][j] = i==j? INIT_VAL: 0;
        }
    }
    return m;
}

void print_matrix(matrix m) {
    for (const auto &l : m) {
        for (const auto &x : l) {
            printf ("%2i ", x);
        }
        printf ("\n");
    }
}

enum op_name {XOR=0, MUL=1, CPY=2, NONE=3};

typedef struct {
    char type;
    char from, to;
} algo_op;

class AlgoState {
    AlgoState * pred;
    algo_op op;
public:
    char weight, weight_to_MDS, nb_mul;

    AlgoState(): AlgoState(NULL, (algo_op){NONE, 0, 0}, 0, NB_INPUTS*XOR_WEIGHT, 0) {};
    AlgoState(AlgoState *_pred, algo_op _op, int _weight, int _weight_to_MDS, int _nb_mul): pred(_pred), op(_op), weight(_weight), weight_to_MDS(_weight_to_MDS), nb_mul(_nb_mul) {};

    matrix branch_vals(bool &overflow);
    matrix branch_vals();
    void print_state (bool maybeMDSwithout[NB_REGISTERS] = NULL, FILE* stream=stdout);
    void print_op (FILE *stream=stdout);
    void spawn_next_states (tbb::concurrent_queue<AlgoState>* remaining_states, matrix_set& scanned_states);
    int queue_weight() const {
        int d = depth();
        assert(d < MAX_DEPTH);
        return (weight + weight_to_MDS)*MAX_DEPTH + d;
    }
    int depth(int *outputs = NULL) const;
    int next_mul() const;
    bool operator <(const AlgoState &other) const {
        return queue_weight() > other.queue_weight();
    }
};

typedef tbb::concurrent_queue<AlgoState> state_queue;
typedef tbb::concurrent_vector<AlgoState> state_vector;
// IMPORTANT: Growing the container does not invalidate existing iterators


void AlgoState::print_op (FILE *stream) {
    if (op.type == NONE)
        return;

    pred->print_op(stream);
    fprintf (stream, "%c (%d, %d)\n", (op.type==XOR?'x':(op.type==MUL?'m':'c')), op.from, op.to);
}


void AlgoState::print_state (bool maybeMDSwithout[NB_REGISTERS], FILE *stream) {
    int i, input_n;
    matrix bv = branch_vals();
    fprintf (stream, "Current state:\n");
    for (i=0; i<NB_REGISTERS; i++) {
        for (input_n=0; input_n<NB_INPUTS; input_n++) {
#ifdef TRY_DIV
            fprintf (stream, "%x.%04x ", bv[i][input_n]/INIT_VAL, bv[i][input_n]%INIT_VAL);
#else
            fprintf (stream, "%x ", bv[i][input_n]);
#endif
        }
        if (maybeMDSwithout && maybeMDSwithout[i])
            fprintf (stream, " [X]");
        fprintf (stream, "\n");
    }
    fprintf (stream, "Operations:\n");
    print_op(stream);
    fprintf (stream, "End of state\n");
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
int test_minors (bool zero, matrix M, bool maybeMDSwithout[NB_REGISTERS] = NULL) {
    /* Note that this function assumes that we have a N x 4 matrix M, and we select 4 lines, yielding a 4 x 4 matrix. */
    
    __m128i dim2det[NB_REGISTERS][NB_REGISTERS][NB_INPUTS][NB_INPUTS];
    __m128i dim3det[NB_REGISTERS][NB_REGISTERS][NB_REGISTERS][NB_INPUTS][NB_INPUTS][NB_INPUTS];
    
    __m128i MM[NB_REGISTERS][NB_INPUTS];

    int line1, line2, line3, column1, column2, column3;    
    int max = 0;

    if (maybeMDSwithout == NULL)
        maybeMDSwithout = (bool*) alloca(NB_REGISTERS*sizeof(bool));

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
    if (zero && !std::accumulate(maybeMDSwithout, maybeMDSwithout+NB_REGISTERS, 0))
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
        if (zero && !std::accumulate(maybeMDSwithout, maybeMDSwithout+NB_REGISTERS, 0))
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

        if (zero && !std::accumulate(maybeMDSwithout, maybeMDSwithout+NB_REGISTERS, 0))
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

        if (zero && !std::accumulate(maybeMDSwithout, maybeMDSwithout+NB_REGISTERS, 0))
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

#define shift(x,n) ((n)>0? shiftl(x,n): shiftr(x,-n))
#define shiftl(x,n) ((n)>=32? 0: ((x)<<(n)))
#define shiftr(x,n) ((n)>=32? 0: ((x)>>(n)))

matrix AlgoState::branch_vals(bool &overflow) {
    overflow = false;
    if (op.type == NONE)
        return init_matrix();
    
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
            bv[op.to][op.from-NB_REGISTERS] ^= INIT_VAL;
        }
    }
    else if (op.type==MUL) {
        for (input_n=0; input_n<NB_INPUTS; input_n++) {
            uint32_t val = bv[op.to][input_n];
            bv[op.to][input_n] = shift(bv[op.to][input_n], op.from); // op.from stores the shift value.
            if (shift(bv[op.to][input_n], -op.from) != val) { // Overflow.
                fprintf (stderr, "Overflow !!\n");
                fprintf (stderr, "Op: MUL(%i,%i)\nPrev:\n", op.from, op.to);
                pred->print_state(NULL, stderr);
                fprintf(stderr, "\n\n");
                overflow = true;
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
            bv[op.to][op.from-NB_REGISTERS] = INIT_VAL;
        }
    }
    return bv;
}

int AlgoState::next_mul() const {
    // Eg, with NB_INPUTS == 2, we need products of two terms for determinant:
    // [1,a,aa,b,ba,baa,bb,bba,bbaa,c,...]
    //  0 1    3                    9

    int res = 1;
    for (int i=0; i<nb_mul; i++)
        res *= (NB_INPUTS+1);
    return res;
}

int AlgoState::depth(int* outputs) const {
    if (op.type == NONE) {
        if (outputs)
            for (int i=0; i<NB_REGISTERS; i++)
                outputs[i] = 0;
        return 0;
    }

    int my_outputs[NB_REGISTERS];
    if (!outputs)
        outputs = my_outputs;
    int d = pred->depth(outputs);
    switch (op.type) {
    case XOR:
        if (op.from < NB_REGISTERS)
            outputs[(int)op.to] = std::max(outputs[(int)op.to], outputs[(int)op.from])+1;
        else
            outputs[(int)op.to]++;
        break;
    case MUL:
        outputs[(int)op.to]++;
        break;
    case CPY:
        if (op.from < NB_REGISTERS)
            outputs[(int)op.to] = outputs[(int)op.from];
        else
            outputs[(int)op.to] = 0;
        break;
    default:
        assert(0);
    }
    int m = *std::max_element(outputs, outputs+NB_REGISTERS);
    return std::max(d,m);
}

matrix AlgoState::branch_vals() {
    bool ignore_overflow;
    return branch_vals(ignore_overflow);
}

bool test_restrictions_MDS (matrix M, bool maybeMDSwithout[NB_REGISTERS] = NULL) {
    return test_minors(true, M, maybeMDSwithout) == NB_INPUTS+1;
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
    int type_of_op_int, to, from;
    enum op_name type_of_op;
   
    for (type_of_op_int=XOR; type_of_op_int<=CPY; type_of_op_int++) {
        // No 2 copies in a row.
        type_of_op = (enum op_name) type_of_op_int;
        if (op.type == CPY && type_of_op == CPY) {
            continue;
        }
#ifdef INDEP_MUL
        if (type_of_op == MUL && next_mul() > 32)
            continue;
#endif

        for (to=0; to<NB_REGISTERS; to++) {
            // After a copy, the copy must be the destination of the operation.
            if (op.type == CPY && op.to != to)
                continue;
            int bound[][3] = {
#ifdef KEEP_INPUTS
                [XOR]={0, NB_INPUTS+NB_REGISTERS, 1},
#else
                [XOR]={0, NB_REGISTERS, 1},
#endif
#ifdef INDEP_MUL
		[MUL]={next_mul(), next_mul()+1, 1},
#else
#ifdef DIFFERENT_MUL
#ifdef TRY_DIV
                [MUL]={-2, 3, 1},
#else
		[MUL]={1, 3, 1},
#endif
#else
#ifdef TRY_DIV
		[MUL]={-1, 2, 2},
#else
                [MUL]={1, 2, 1},
#endif
#endif
#endif
#ifdef KEEP_INPUTS
                [CPY]={0, NB_INPUTS+NB_REGISTERS, 1},
#else
                [CPY]={0, NB_REGISTERS, 1},
#endif
            };
            for (from=bound[type_of_op][0]; from<bound[type_of_op][1]; from+=bound[type_of_op][2]) {
                if (type_of_op!=MUL && to==from)
                    continue;
                int new_weight = weight + (type_of_op==XOR?XOR_WEIGHT:(type_of_op==MUL?MUL_WEIGHT:CPY_WEIGHT));
                if (new_weight >= MAX_WEIGHT)
                    continue;

                AlgoState next_state(this, (algo_op){type_of_op, (char)from, (char)to}, new_weight, 0, nb_mul+(type_of_op==MUL));
                if (next_state.depth() >= MAX_DEPTH)
                    continue;
                
                /* We filter here 2 things:
                 *  - injectivity (can only change after a copy).
                 *  - equivalence up to input/output reordering (id).
                 * Note that the id test at this point can only test if we have already SCANNED a state with the same id as next_state.
                 * We cannot test if another state with the same id is in the queue since we don't store the id in the queue (to save memory).
                 */
                bool overflow;
                matrix bv = next_state.branch_vals(overflow);
                if (overflow)
                    continue;
                if (type_of_op == CPY) {
                    // Injectivity test assumes that NB_REGISTERS = NB_INPUTS+1;
#ifndef KEEP_INPUTS
                    if (rank(bv) != NB_INPUTS) {
                        continue;
                    }
#endif
                }
#ifdef COMPUTE_ID_FIRST
                matrix id = compute_id(bv);
                if (scanned_states.find(id) != scanned_states.end()) { // Id already scanned.
                    continue;
                }
#endif                
                // Computing a bound on the distance to an MDS matrix.
                next_state.weight_to_MDS = min_dist_to_MDS(bv);
                
                assert(next_state.queue_weight() >= queue_weight());
                if (next_state.queue_weight() < MAX_QUEUE_WEIGHT) {
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

typedef struct {
    state_queue *remaining_states;
    state_vector* scanned_states;
    matrix_set *scanned_ids;
    
} mem_thr_param;

void* memory_printer(void *p) {
    mem_thr_param *param = (mem_thr_param*) p;
    struct rusage usage;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    time_t init_time = ts.tv_sec;
    for (;;) {
        sleep(SLEEP_TIME);
        getrusage(RUSAGE_SELF, &usage);
        clock_gettime(CLOCK_MONOTONIC, &ts);
        unsigned long long remaining1 = 0;
        unsigned long long remaining = 0;
        int first = -1;
        for (int ii=0; ii<MAX_QUEUE_WEIGHT; ii++) {
#ifdef DEPTH_FIRST
            int i = (ii/MAX_WEIGHT) + (ii%MAX_WEIGHT)*MAX_DEPTH;
#else
            int i = ii;
#endif
            size_t s = param->remaining_states[i].unsafe_size();
            if (s && first == -1)
                first=i, remaining1 = s;
            remaining += s;
        }
        fprintf (stderr, "Time: %6llus, Mem: %4.3fGB - scanned: %11llu [%11llu] - remaining: %11llu [%11llu@%i.%i] - user time: %llus\n",
                 (unsigned long long) ts.tv_sec - init_time, (double)usage.ru_maxrss/1024/1024,
                 (unsigned long long) param->scanned_states->size(), (unsigned long long) param->scanned_ids->size(),
                 remaining, remaining1, first/MAX_DEPTH, first%MAX_DEPTH, (unsigned long long) usage.ru_utime.tv_sec);
        fflush(stderr);
    }
    return NULL;
}

#define PRINT_PARAM(x) #x "=" stringify(x)
int main () {
    printf ("Parameters:"
            " " PRINT_PARAM(MAX_WEIGHT)
            " " PRINT_PARAM(MAX_DEPTH)
            " " PRINT_PARAM(NB_INPUTS)
            " " PRINT_PARAM(NB_REGISTERS)
            " " PRINT_PARAM(XOR_WEIGHT)
            " " PRINT_PARAM(MUL_WEIGHT)
            " " PRINT_PARAM(CPY_WEIGHT)
            "\n");
    printf ("Size: AlgoState:%i algo_op:%i matrix:%i\n",
            (int)sizeof(AlgoState), (int)sizeof(algo_op),
            (int)sizeof(matrix));
    printf ("Options:"
#ifdef DEPTH_FIRST
            " DEPTH_FIRST"
#endif
#ifdef KEEP_INPUTS
            " KEEP_INPUTS"
#endif
#ifdef TRY_DIV
            " TRY_DIV"
#endif
#ifdef DIFFERENT_MUL
            " DIFFERENT_MUL"
#endif
#ifdef INDEP_MUL
            " INDEP_MUL"
#endif
            "\n");

    state_queue remaining_states[MAX_QUEUE_WEIGHT];
    state_vector scanned_states;
    matrix_set scanned_ids;

    pthread_t mem_thr;
    mem_thr_param param = {remaining_states, &scanned_states, &scanned_ids};
    pthread_create(&mem_thr, NULL, memory_printer, &param);

    AlgoState initial_state;
    remaining_states[0].push(initial_state);
    
    uint64_t nb_scanned = 0, nb_tested = 0;

    try {
        for (int cw=0; cw<MAX_QUEUE_WEIGHT; cw++) {
#ifdef DEPTH_FIRST
            int current_weight = (cw/MAX_WEIGHT) + (cw%MAX_WEIGHT)*MAX_DEPTH;
#else
            int current_weight = cw;
#endif
            printf ("New weight : %d.%d\n", current_weight/MAX_DEPTH, current_weight%MAX_DEPTH);
            printf ("Number of distinct ids : %" PRIu64 "/%" PRIu64 "(1/%.1f)\n", nb_scanned, nb_tested, (nb_scanned?(float)nb_tested/nb_scanned:0));
            printf ("Scanned size : %lu\n", scanned_ids.size());
            //                printf ("Remaining size : %lu\n", remaining_states[w].size());
            fflush(stdout);
        
            uint64_t old_nb_scanned = nb_scanned;
            nb_scanned = 0;
            uint64_t old_nb_tested = nb_tested;
            nb_tested = 0;
        
#pragma omp parallel reduction(+:nb_scanned) reduction(+:nb_tested)
            {
                // Next function to test.
                AlgoState current_state;

                while (remaining_states[current_weight].try_pop(current_state)) {
                    nb_tested++;
                    matrix bv = current_state.branch_vals();
                    matrix id = compute_id(bv); // Get a unique id invariant under input/output reordering.
                    bool is_new;
                    try {
                        is_new = scanned_ids.insert(id).second;
                    } CATCH;
                    if (is_new) { // Current state not scanned yet (even up to input/output reordering).
                        nb_scanned++;
                        bool maybeMDSwithout[NB_REGISTERS];
                        if (current_state.weight_to_MDS == 0 && test_restrictions_MDS(bv, maybeMDSwithout)) { // Test if any restriction to 4 output branches is MDS. If so, prints and ends.
#pragma omp critical
                            {
                                printf ("Found MDS !!! (weight:%i) (depth:%i)\n", current_state.weight, current_state.depth());
                                // printf ("Number of distinct ids (weight=%3d, depth=%3d) : %" PRIu32 "/%" PRIu32 "(1/%.1f)\n",
                                //         current_weight, current_state.depth(), nb_scanned, nb_tested, (nb_scanned?(float)nb_tested/nb_scanned:0));
                                current_state.print_state(maybeMDSwithout);
                                fflush(stdout);
                            }
                        }
                        if (current_state.weight == MAX_WEIGHT-1)
                            continue; // If it's not MDS, adding free operations will not make it MDS
                        
                        // Checking id.
                        // Insert into scanned_states
                        state_vector::iterator tmp;
                        try {
                            tmp = scanned_states.push_back(current_state);
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
    sleep(SLEEP_TIME);
    //exit(0);
}

/* Local Variables: */
/* indent-tabs-mode: nil */
/* c-basic-offset: 4 */
/* End: */
