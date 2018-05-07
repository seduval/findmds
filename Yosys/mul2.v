// File ../../AES/AES_mul2.vhd translated with vhd2vl v2.5 VHDL to Verilog RTL translator
// vhd2vl settings:
//  * Verilog Module Declaration Style: 1995

// vhd2vl is Free (libre) Software:
//   Copyright (C) 2001 Vincenzo Liguori - Ocean Logic Pty Ltd
//     http://www.ocean-logic.com
//   Modifications Copyright (C) 2006 Mark Gonzales - PMC Sierra Inc
//   Modifications (C) 2010 Shankar Giri
//   Modifications Copyright (C) 2002, 2005, 2008-2010, 2015 Larry Doolittle - LBNL
//     http://doolittle.icarus.com/~larry/vhd2vl/
//
//   vhd2vl comes with ABSOLUTELY NO WARRANTY.  Always check the resulting
//   Verilog for correctness, ideally with a formal verification tool.
//
//   You are welcome to redistribute vhd2vl under certain conditions.
//   See the license (GPLv2) file included with the source for details.

// The result of translation follows.  Its copyright status should be
// considered unchanged from the original VHDL.

//-----------------------------------------------------------------------------
//! @brief  Multiplication by 2 (NOT A FIELD)
//-----------------------------------------------------------------------------
// no timescale needed

module mul2 (a, b);

input [7:0] a;
output [7:0] b;


  assign b[7] = a[6];
  assign b[6] = a[5];
  assign b[5] = a[4];
  assign b[4] = a[3];
  assign b[3] = a[2];
  assign b[2] = a[7] ^ a[1];
  assign b[1] = a[0];
  assign b[0] = a[7];

endmodule
