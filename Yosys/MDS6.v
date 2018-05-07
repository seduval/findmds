// File ../../AES/AES_MixColumn.vhd translated with vhd2vl v2.5 VHDL to Verilog RTL translator
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
//! @file       AES_MixColumn.vhd
//! @brief      A single operation of MixColumns operation
//! @project    CAESAR Candidate Evaluation
//! @author     Marcin Rogawski   
//! @author     Ekawat (ice) Homsirikamol
//! @copyright  Copyright (c) 2014 Cryptographic Engineering Research Group
//!             ECE Department, George Mason University Fairfax, VA, U.S.A.
//!             All rights Reserved.
//! @license    This project is released under the GNU Public License.
//!             The license and distribution terms for this file may be
//!             found in the file LICENSE in this distribution or at 
//!             http://www.gnu.org/licenses/gpl-3.0.txt
//! @note       This is publicly available encryption source code that falls
//!             under the License Exception TSU (Technology and software-
//!             —unrestricted)
//-----------------------------------------------------------------------------
// no timescale needed

module MDS6(a, b, c, d, s, t, u, v);

input [7:0] a;
input [7:0] b;
input [7:0] c;
input [7:0] d;
output [7:0] s;
output [7:0] t;
output [7:0] u;
output [7:0] v;

//-----------------------------------------------------------------------------
//! @brief  NOT THE AES!!!
//-----------------------------------------------------------------------------
wire [7:0] sum1;
wire [7:0] sum2;
wire [7:0] sum3;
wire [7:0] sum4;
wire [7:0] sum5;
wire [7:0] sum6;
wire [7:0] sum7;
wire [7:0] sum8;

wire [7:0] mul1;
wire [7:0] mul2;
wire [7:0] mul3;

  assign sum1 = a^b;
  assign sum2 = c^d;
  mul2 m1(sum1, mul1);
  assign sum3 = d^mul1;
  assign sum4 = b^sum2;
  mul2 m2(sum4, mul2);
  mul2 m3(sum3, mul3);
  assign sum5 = sum2^mul3;
  assign sum6 = sum1^mul2;
  assign sum7 = mul2^sum5;
  assign sum8 = sum6^sum3;

  assign s = sum6;
  assign t = sum7;
  assign u = sum5;
  assign v = sum8;

endmodule
