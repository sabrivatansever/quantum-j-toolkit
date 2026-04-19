import sympy as sp
import numpy as np
import re

# -------------------------------
# GLOBALS
# -------------------------------
hbar = 1
theta, phi = sp.symbols('theta phi', real=True)

# -------------------------------
# MATRIX REPRESENTATION
# -------------------------------
def j_matrices(j):
    dim = int(2*j + 1)
    m_vals = [j - i for i in range(dim)]

    Jz = sp.zeros(dim)
    Jp = sp.zeros(dim)
    Jm = sp.zeros(dim)

    for i, m in enumerate(m_vals):
        Jz[i, i] = m

        if i < dim - 1:
            coef = sp.sqrt(j*(j+1) - m*(m-1))
            Jm[i+1, i] = coef
            Jp[i, i+1] = coef

    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / (2*sp.I)

    return Jx, Jy, Jz, Jp, Jm


# -------------------------------
# LADDER ACTION
# -------------------------------
def apply_ladder(j, m, operator):
    if operator == "J+":
        if m >= j:
            return 0, None
        coef = sp.sqrt(j*(j+1) - m*(m+1))
        return coef, m+1

    elif operator == "J-":
        if m <= -j:
            return 0, None
        coef = sp.sqrt(j*(j+1) - m*(m-1))
        return coef, m-1

    elif operator == "Jz":
        return m, m

    return None, None


# -------------------------------
# OPERATOR CHAIN
# -------------------------------
def apply_operator_chain(expr, j, m):
    ops = expr.strip().split()

    coef = 1
    current_m = m

    for op in reversed(ops):
        c, new_m = apply_ladder(j, current_m, op)

        if c == 0:
            return 0, None

        coef *= c
        current_m = new_m

    return sp.simplify(coef), current_m


# -------------------------------
# COMMUTATOR
# -------------------------------
def commutator(A, B):
    return A*B - B*A


def apply_commutator(expr, j, m):
    match = re.match(r"\[(.*),(.*)\]", expr)
    if not match:
        return None

    A = match.group(1).strip()
    B = match.group(2).strip()

    coef1, m1 = apply_operator_chain(f"{A} {B}", j, m)
    coef2, m2 = apply_operator_chain(f"{B} {A}", j, m)

    if m1 == m2:
        return sp.simplify(coef1 - coef2), m1

    return "Different resulting states"


# -------------------------------
# DIFFERENTIAL OPERATORS
# -------------------------------
def Jz_op(psi):
    return -sp.I * sp.diff(psi, phi)


def J2_op(psi):
    return -(
        (1/sp.sin(theta)) *
        sp.diff(sp.sin(theta) * sp.diff(psi, theta), theta)
        + (1/sp.sin(theta)**2) * sp.diff(psi, phi, 2)
    )


# -------------------------------
# SPHERICAL HARMONICS
# -------------------------------
def spherical_harmonic(l, m):
    return sp.simplify(sp.functions.special.spherical_harmonics.Ynm(l, m, theta, phi))


# -------------------------------
# DIRAC PARSER
# -------------------------------
def parse_input(user_input):
    parts = user_input.split("|")

    if len(parts) != 2:
        return None

    ops_part = parts[0].strip()
    state_part = parts[1].replace(">", "").strip()

    j, m = map(float, state_part.split(","))

    return ops_part, j, m


# -------------------------------
# MAIN PROGRAM
# -------------------------------
def main():
    print("=== Quantum J Toolkit ===")

    print("""
1) Matrix representation
2) Ladder operator on |j,m>
3) Commutator (matrix)
4) Dirac expression (e.g. J+ J- |1,0>)
5) Differential operators on ψ
6) Spherical harmonics Y_l^m
""")

    choice = input("Choice: ")

    # ---------------------------
    if choice == "1":
        j = float(input("Enter j: "))
        Jx, Jy, Jz, Jp, Jm = j_matrices(j)

        print("\nJx ="); sp.pprint(Jx)
        print("\nJy ="); sp.pprint(Jy)
        print("\nJz ="); sp.pprint(Jz)
        print("\nJ+ ="); sp.pprint(Jp)
        print("\nJ- ="); sp.pprint(Jm)

    # ---------------------------
    elif choice == "2":
        j = float(input("j: "))
        m = float(input("m: "))
        op = input("Operator (J+ or J-): ")

        coef, new_m = apply_ladder(j, m, op)

        if coef == 0:
            print("\nResult: 0")
        else:
            print(f"\nResult: {coef} |{j},{new_m}>")

    # ---------------------------
    elif choice == "3":
        j = float(input("j: "))
        Jx, Jy, Jz, Jp, Jm = j_matrices(j)

        ops = {"Jx": Jx, "Jy": Jy, "Jz": Jz, "Jp": Jp, "Jm": Jm}

        A = input("First operator: ")
        B = input("Second operator: ")

        result = commutator(ops[A], ops[B])

        print("\nResult:\n")
        sp.pprint(sp.simplify(result))

    # ---------------------------
    elif choice == "4":
        expr = input("Enter expression (e.g. J+ |1,0>): ")
        parsed = parse_input(expr)

        if not parsed:
            print("Invalid format")
            return

        ops, j, m = parsed

        if ops.startswith("["):
            result = apply_commutator(ops, j, m)
            if isinstance(result, tuple):
                coef, new_m = result
                print(f"\nResult: {coef} |{j},{new_m}>")
            else:
                print(result)
        else:
            coef, new_m = apply_operator_chain(ops, j, m)
            print(f"\nResult: {coef} |{j},{new_m}>")

    # ---------------------------
    elif choice == "5":
        psi_input = input("ψ(θ,φ) = ")
        psi = sp.sympify(psi_input)

        print("\nJz ψ =")
        sp.pprint(Jz_op(psi))

        print("\nJ² ψ =")
        sp.pprint(J2_op(psi))

    # ---------------------------
    elif choice == "6":
        l = int(input("l: "))
        m = int(input("m: "))

        Y = spherical_harmonic(l, m)

        print("\nY_l^m =")
        sp.pprint(Y)

        print("\nCheck Jz eigenvalue:")
        sp.pprint(Jz_op(Y))

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
