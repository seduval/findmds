#include <stdio.h>
#include <assert.h>
#include <string.h>

#define WORDSIZE 8
#define TRUNC(x) ((x) & ((1<<WORDSIZE)-1))
#define ROL(x) TRUNC(((x)<<1)|((x)>>(WORDSIZE-1)))
#define ROR(x) TRUNC(((x)>>1)|((x)<<(WORDSIZE-1)))

#if WORDSIZE == 8
#define MUL(x) (ROL(x) ^ (((x)>>1)&1))
#define DIV(x) ROR((x) ^ (((x)>>2)&1))
#else
#define MUL(x) (ROL(x) ^ ((x)&1))
#define DIV(x) (ROR(x) ^ TRUNC(ROR(x)<<(WORDSIZE-1)))
#endif

typedef struct {
  int x[4];
} state;

state f(state x) {
  x.x[0] ^= x.x[1];
  x.x[2] ^= x.x[3];
  x.x[1] ^= x.x[2];

  x.x[2] ^= MUL(x.x[0]);
  x.x[1]  = MUL(x.x[1]);
  x.x[3] ^= x.x[1];
  x.x[3]  = MUL(x.x[3]);
  x.x[0] ^= x.x[3];
  x.x[3] ^= x.x[2];
  x.x[1] ^= x.x[0];
  
  return x;
}

#ifdef DIV
state g(state x) {
  x.x[1] ^= x.x[0];
  x.x[3] ^= x.x[2];
  x.x[0] ^= x.x[3];
  x.x[3]  = DIV(x.x[3]);
  x.x[3] ^= x.x[1];
  x.x[1]  = DIV(x.x[1]);
  x.x[2] ^= MUL(x.x[0]);

  x.x[1] ^= x.x[2];
  x.x[2] ^= x.x[3];
  x.x[0] ^= x.x[1];
  
  return x;
}
#endif

int w(state x) {
  return (!!x.x[0]) + (!!x.x[1]) + (!!x.x[2]) + (!!x.x[3]);
}

int main () {
  for (int a = 0; a < 1<<WORDSIZE; a++)
    for (int b = 0; b < 1<<WORDSIZE; b++)
      for (int c = 0; c < 1<<WORDSIZE; c++) {
	for (int d = 0; d < 1<<WORDSIZE; d++) {
	  state x = {{a, b, c, d}};
	  state y = f(x);
	  if (w(y)==0) {
	    printf ("# y = 0: %x %x %x %x\n", a, b, c, d);
	  }
	  if (w(x) + w(y) < 5) {
	    printf ("! w = %i: %x %x %x %x\n", w(x) + w(y), a, b, c, d);
	  }
#ifdef DIV
	  state z = g(y);
	  assert (memcmp(&x, &z, sizeof(x)) == 0);
#endif
	}
      }
}
