load("code.sage")
import re
import os
import subprocess

RR = PolynomialRing(GF(2), name="x")

R8 = RR.quotient_ring("(x^4 + x + 1)^2")
A8 = R8("x")

R4 = RR.quotient_ring("(x^4 + x + 1)")
A4 = R4("x")

F8 = GF(2^8)
AA = F8.gen()

lightweight_circuits = []

M4593 = """
x1 += x2
x3 += x4
t   = x1
t   = A*t
x4 += t
x2 += x3
x2  = A*x2
x1 += x2
x2 += x3
t   = x4
t   = A*t
x2 += t
x3 += t
x4 += x1
"""
M4683 = """
x1 += x2
x3 += x4
t   = x1
t   = A*t
x4 += t
x2 += x3
x2 = A*x2
x1 += x2
t   = x4
t   = A*t
x3 += t
x2 += x3
x4 += x1
"""
M4583 = """
x1 += x2
x3 += x4
x4 += x1
x2 += x3
x3 = (A**(-1))*x3
x2 = (A**2)*x2
x1 = A*x1
x1 += x2
x3 += x4
x4 += x1
x2 += x3
"""
M4494 = """
x2 += x4
x4 += x1
x1 += x3
x1 += x2
x3 = A*x3
x3 += x2
t   = x1
t   = A*t
x2 += t
x3 = A*x3
t   = x4
t   = A*t
x1 += t
x4 += x3
x3 += x1
"""
M4493 = """
x1 += x2
x2 += x3
x3 += x4
x4 += x1
x1 = A*x1
x3 = (A**2)*x3
x1 += x2
t = x4
t = (A**(-1))*t
x2 += t
x4 += x3
x3 += x1
x1 += x4
"""
M4484a = """
x1 += x2
x3 += x4
x2 = A*x2
x4 = A*x4
x2 += x3
x4 += x1
x3 = (A**2)*x3
x1 = (A**2)*x1
x3 += x4
x1 += x2
x4 += x1
x2 += x3
"""
M4484b = """
x1 += x2
x3 += x4
x2 += x3
x4 += x1
x1 = A*x1
x3 = A*x3
x3 += x4
x1 += x2
x4 = (A**2)*x4
x2 = (A**2)*x2
x4 += x1
x2 += x3
"""
M4484c = """
x1 += x2
x3 += x4
x4 = A*x4
x2 += x3
x4 += x1
x3 = (A**2)*x3
x1 = A*x1
x3 += x4
x1 += x2
x2 = A*x2
x4 += x1
x2 += x3
"""
M4395 = """
t = x1
t = (A**(-1))*t
x1 += x2
x2 += x3
x3 += x4
x3 += t
t = x2
t = (A**(-1))*t
x2 = A*x2
x4 = (A**(-1))*x4
t += x3
x4 += x1
x2 += x4
x4 += x3
x1 = A*x1
x1 += x3
x3 = t
"""
M4395b = """
t = x1
t += x4
x1 = A*x1
x1 += x2
x1 += x3
x2 += x4
x4 = x2
x3 = A*x3
x3 += t
t = (A**(-1))*t
x2 = A*x2
x4 = (A**(-1))*x4
x2 += x1
x4 += x3
x3 += x1
x1 += t
"""
M4484d = """
t = x2
t += x4
x2 = t
x2 = (A**2)*x2
x1 += x3
x4 += x1
x2 += x4
x1 = (A**2)*x1
x3 += t
x1 += x3
x3 = A*x3
x3 += x2
x4 = A*x4
x4 += x1
"""
M4583b = """
x1 += x2
x3 += x4
x2 += x3
x2 = A*x2
x4 += x1
x1 += x2
x3 = (A**2)*x3
x4 = (A**(-1))*x4
x3 += x4
x2 += x3
x4 += x1
"""

lightweight_circuits.extend(
    [("M4593_4" , M4593, R4, A4),
     ("M4593_4b", M4593, R4, A4^-1),
     ("M4593_8" , M4593, R8, A8),
     ("M4593_8b", M4593, R8, A8^-1)]
)

lightweight_circuits.extend(
    [("M4683_4" , M4683, R4, A4),
     ("M4683_4b", M4683, R4, A4^-1),
     ("M4683_8" , M4683, R8, A8),
     ("M4683_8b", M4683, R8, A8^-1)]
)

lightweight_circuits.extend(
    [("M4583_4" , M4583, R4, A4^-1),
     ("M4583_8" , M4583, R8, A8^-1)]
)

lightweight_circuits.extend(
    [("M4494_4" , M4494, R4, A4),
     ("M4494_8" , M4494, R8, A8)]
)

lightweight_circuits.extend(
    [("M4493_4" , M4493, R4, A4),
     ("M4493_8" , M4493, R8, A8)]
)

lightweight_circuits.extend(
    [("M4484a_4" , M4484a, R4, A4),
     ("M4484a_8" , M4484a, R8, A8)]
)

lightweight_circuits.extend(
    [("M4484b_4" , M4484b, R4, A4),
     ("M4484b_8" , M4484b, R8, A8)]
)

lightweight_circuits.extend(
    [("M4484c_4" , M4484c, R4, A4),
     ("M4484c_8" , M4484c, R8, A8)]
)

lightweight_circuits.extend(
    [("M4395_4" , M4395, R4, A4),
     ("M4395_4b", M4395, R4, A4^-1),
     ("M4395_8" , M4395, R8, A8),
     ("M4395_8b", M4395, R8, A8^-1)]
)

lightweight_circuits.extend(
    [("M4395b_4" , M4395b, R4, A4),
     ("M4395b_4b", M4395b, R4, A4^-1),
     ("M4395b_8" , M4395b, R8, A8),
     ("M4395b_8b", M4395b, R8, A8^-1)]
)

#New
lightweight_circuits.extend(
    [("M4484d_4" , M4484d, R4, A4),
     ("M4484d_8" , M4484d, R8, A8)]
)

lightweight_circuits.extend(
    [("M4583b_4" , M4583b, R4, A4),
     ("M4583b_8" , M4583b, R8, A8)]
)


def circuit2matrix(program, R, A):
    """
    Converts word-based circuit into matrix
    """
    function = """def prog(R,A):
    # Define initial state
    x1 = vector(R,[1,0,0,0])
    x2 = vector(R,[0,1,0,0])
    x3 = vector(R,[0,0,1,0])
    x4 = vector(R,[0,0,0,1])
    t  = vector(R,[0,0,0,0])

    # Compute program
    {:s}

    # Output matrix
    return matrix([x1,x2,x3,x4])
    """.format("\n    ".join(program.splitlines()))
    exec(function)
    return prog(R,A)

def is_MDS(M):
    can_invert = lambda m: m.is_unit()
    minors = [ m for k in range (M.nrows()) for m in M.minors(k+1) ]
    return all(can_invert(m) for m in minors)

def check_slp_program(m, program, offset=0):
    """
    Return whether the encoded program implements the multiplication by m
    """
    if m[0][0].parent() != GF(2):
        m = ff_matrix2bin(m)

    # define program
    n_variables = m.nrows()
    inputs = ["x%d" % (i + offset) for i in range(n_variables)]
    inputs2gf2 = ["%s = GF(2)(%s)" % (i, i) for i in inputs]
    outputs = ["y%d" % (i + offset) for i in range(n_variables)]
    function = """def prog(x):
    # parse input to variables
    %s = x

    # convert variables to GF(2) objects
    %s

    # compute given program
    %s

    return vector(GF(2), [%s])
    """ % (", ".join(inputs),
           "\n    ".join(inputs2gf2),
           "\n    ".join(program.splitlines()),
           ", ".join(outputs))
    exec(function)

    test_m = [prog(Integer(1<<i).digits(base=2, padto=n_variables))
              for i in range(n_variables)
             ]
    test_m = matrix(GF(2), test_m).transpose()
    return m == test_m

def mul2vhdl(A, str):
    """
    Converts ring operation into VHDL code
    """
    n = A.parent().degree()
    S = """
    module {1}(a,b);

    input  [{0}:0] a;
    output [{0}:0] b;
    \n""".format(n-1, str)

    M = field2bin(A);
    for i in range(n):
        S += "    assign b[{}] = ".format(i)+(" ^ ".join([ "a[{}]".format(j) for j in field2bin(A)[i].support() ]))+";\n"
    S += """

    endmodule
    """
    return S

def circuit_cost(program, R, A):
    """
    Evaluate cost of circuit
    """
    n = R.degree()
    m1 = naive_xor_count(matrix(R,[[A]]))
    i1 = naive_xor_count(matrix(R,[[A^-1]]))
    depth = {"x1": ([0]*n), "x2": ([0]*n), "x3": ([0]*n), "x4": ([0]*n)}
    def mul(l,M):
        M = field2bin(M)
        ll = copy(l)
        for i in range(n):
            w = len(M[i].support())
            l[i] = ceil(log(w)/log(2)) + max([ll[j] for j in M[i].support()])
    C = 0
    for l in program.splitlines():
        match_xor = re.search(r"^([xt][1-4]?) *\+= *([xt][1-4]?)$", l)
        match_cpy = re.search(r"^([xt][1-4]?) *= *([xt][1-4]?)$", l)
        match_mul1 = re.search(r"^([xt][1-4]?) *= *A\*\1$", l)
        match_mul2 = re.search(r"^([xt][1-4]?) *= *\(A\*\*2\)\*\1$", l)
        match_div1 = re.search(r"^([xt][1-4]?) *= *\(A\*\*\(-1\)\)\*\1$", l)
        match_div2 = re.search(r"^([xt][1-4]?) *= *\(A\*\*\(-2\)\)\*\1$", l)
        match_empty = re.search(r"^ *$", l)
        if match_xor:
            C += n
            depth[match_xor.group(1)] = [ 1 + max(depth[match_xor.group(1)][i], depth[match_xor.group(2)][i]) for i in range(n) ]
        elif match_cpy:
            C += 0
            depth[match_cpy.group(1)] = copy(depth[match_cpy.group(2)])
        elif match_mul1:
            C += m1
            mul(depth[match_mul1.group(1)],A)
        elif match_mul2:
            C += 2*m1
            mul(depth[match_mul2.group(1)],A)
            mul(depth[match_mul2.group(1)],A)
        elif match_div1:
            C += i1
            mul(depth[match_div1.group(1)],A^-1)
        elif match_div2:
            C += 2*i1
            mul(depth[match_div2.group(1)],A^-1)
            mul(depth[match_div2.group(1)],A^-1)
        elif match_empty:
            None
        else:
            raise Exception("Invalid line: "+l)
    return (C, max(sum(depth.values(),[])))
        
def circuit2vhdl(program, R, A):
    """
    Converts word-based circuit into VHDL code
    """
    n = R.degree()
    S = mul2vhdl(A,"mul1")+mul2vhdl(A^2,"mul2")+mul2vhdl(A^-1,"div1")+mul2vhdl(A^-2,"div2")
    S += """
    module matrix(a,b,c,d,s,t,u,v);

    input [{0}:0] a;
    input [{0}:0] b;
    input [{0}:0] c;
    input [{0}:0] d;

    output [{0}:0] s;
    output [{0}:0] t;
    output [{0}:0] u;
    output [{0}:0] v;
    \n""".format(n-1)
    state = {"x1": "a", "x2": "b", "x3": "c", "x4": "d", "t": "0"}
    next_t = 0
    next_l = 0
    lin_dict = { "A": "mul1", "(A**2)": "mul2", "(A**(-1))": "div1", "(A**(-2))": "div2" }
    for l in program.splitlines():
        match_xor = re.search(r"^([xt][1-4]?) *\+= *([xt][1-4]?)$", l)
        match_cpy = re.search(r"^([xt][1-4]?) *= *([xt][1-4]?)$", l)
        match_lin = re.search(r"^([xt][1-4]?) *= *(A|\(A\*\*2\)|\(A\*\*\(-1\)\)|\(A\*\*\(-2\)\))\*\1$", l)
        match_empty = re.search(r"^ *$", l)
        if match_xor:
            next_t += 1
            S += "    wire [{0}:0] tmp{1};\n".format(n-1, next_t)
            S += "    assign tmp{0} = {1}^{2};\n".format(next_t, state[match_xor.group(1)], state[match_xor.group(2)])
            state[match_xor.group(1)] = "tmp{0}".format(next_t)
        elif match_cpy:
            state[match_cpy.group(1)] = state[match_cpy.group(2)]
        elif match_lin:
            next_t += 1
            next_l += 1
            S += "    wire [{0}:0] tmp{1};\n".format(n-1, next_t)
            S += "    {3} l{0}({1}, tmp{2});\n".format(next_l, state[match_lin.group(1)], next_t, lin_dict[match_lin.group(2)])
            state[match_lin.group(1)] = "tmp{0}".format(next_t)
        elif match_empty:
            None
        else:
            raise Exception("Invalid line: "+l)
    S += """
    assign s = {0};
    assign t = {1};
    assign u = {2};
    assign v = {3};

    endmodule
    """.format(state["x1"], state["x2"], state["x3"], state["x4"])
    return S

def yosys(C, R, A, debug=false):
    vhdl = circuit2vhdl(C, R, A)

    vhdl_rfd, vhdl_wfd = os.pipe()
    script_rfd, script_wfd = os.pipe()
    p = subprocess.Popen(
        ["yosys"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=False,                           # make sure file descriptors are kept open in subprocess
        preexec_fn = lambda: (os.close(vhdl_wfd),os.close(script_wfd))    # make sure write end is closed in child
    )
    p.stdin.write("read_verilog \"/dev/fd/{}\"\n".format(vhdl_rfd))
    os.write(vhdl_wfd,vhdl)
    os.close(vhdl_wfd)
    p.stdin.write("hierarchy -check -top matrix\n")
    p.stdin.write("flatten\n")
    p.stdin.write("proc; opt; fsm; opt; memory; opt\n")
    p.stdin.write("techmap; opt\n")
    p.stdin.write("dfflibmap -liberty osu05_stdcells.lib\n")
    p.stdin.write("abc -liberty osu05_stdcells.lib -g XOR\n")
    p.stdin.write("clean\n")
    p.stdin.write("stat\n")
    p.stdin.write("write_blif\n")
    p.stdin.close()

    out = p.stdout.read()
    err = p.stderr.read()
    if debug:
        print out
        print err

    n = A.parent().degree()
    d = {}
    t = [0] # Python closure sucks!!!!!
    SLP = []
    slp = ""
    def parse_var(s):
        x = re.search(r'^([a-d])\[([0-9]+)\]$', s)
        if x:
            return "x{}".format((ord(x.group(1))-ord('a'))*n+int(x.group(2)))
        x = re.search(r'^([s-v])\[([0-9]+)\]$', s)
        if x:
            return "y{}".format((ord(x.group(1))-ord('s'))*n+int(x.group(2)))
        if s in d:
            return d[s]
        else:
            t[0] = t[0]+1
            d[s] = "t{}".format(t[0])
            return d[s]
        
    for l in out.splitlines():
        m = re.search(r'^\.subckt.*A=(.*) B=(.*) Y=(.*)', l)
        if m:
            SLP.append((parse_var(m.group(1)), parse_var(m.group(2)), parse_var(m.group(3))))

    depth = {}
    for i in range(4*n):
        depth["x{}".format(i)] = 0
    while SLP:
        for (a,b,y) in SLP:
            if a in depth and b in depth:
                slp += "{} = {} + {}\n".format(y,a,b)
                depth[y] = 1+max(depth[a],depth[b])
                SLP.remove((a,b,y))
                break
        else:
            raise Exception("Bad Yosys output!")

    # print slp
        
    assert check_slp_program(M, slp)
    assert int(re.search(r'Number of cells: *([0-9]+)', out).group(1)) == len(slp.splitlines())

    return (len(slp.splitlines()), max(depth.values()))
   
def test_matrix(M):
    print "Naive Xor Count: {}".format(naive_xor_count(M))
    print "Paar1          : {}".format(paar1(M))
    # SLP quite slow for big matrices, only run on small matrices for now
    if M[0][0].parent().degree() < 8:
        print "SLP_heuristic  : {}".format(slp_heuristic(M))


for x in lightweight_circuits:
    (str, C, R, A) = x
    print "Testing {}".format(str)
    M = circuit2matrix(C, R, A)
#    matrix2columns(M, to_file="./matrices_paar_header/"+str+".h")
    print M
    assert is_MDS(M)
    print "Ours:          : {}".format(circuit_cost(C, R, A))
    test_matrix(M)
    print "Yosys:         : {}".format(yosys(C, R, A))
    print "\n\n"
