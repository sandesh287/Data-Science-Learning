# Verify properties of matrix multiplication (Eg. A.B != B.A)
# 1. Associative: (AB)C = A(BC)
# 2. Commutative: AB != BA
# 3. Distributive: A(B + C) = AB + AC  ;  (B + C)A = BA + CA
# 4. Multiplicative identity: IA = AI = A
# 5. Multiplicative zero: 0A = A0 = 0
# 6. Dimension property: product of m x n and n x k is m x k matrix

import numpy as np

# create matrices
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
C = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
I = np.eye(3)  # identity matrix
Z = np.zeros([3, 3])  # zero matrix

# Dot products
DotAB = np.dot(A, B)
DotBA = np.dot(B, A)
DotBC = np.dot(B, C)
DotCB = np.dot(C, B)
DotAC = np.dot(A, C)
DotCA = np.dot(C, A)
DotAI = np.dot(A, I)
DotIA = np.dot(I, A)
DotZA = np.dot(Z, A)
DotAZ = np.dot(A, Z)

# Addition
AddAB = A + B
AddAC = A + C
AddBC = B + C
AddBA = B + A
AddCA = C + A
AddCB = C + B

# 1. Associative property: (AB)C = A(BC)
DotAB_C = np.dot(DotAB, C)
DotA_BC = np.dot(A, DotBC)
print(f"Checking Associative Property of \n {DotAB_C} and \n {DotA_BC}")

# using np.allclose() for comparison, as DotAB_C == DotA_BC doesn't work, because NumPy doesn't allow using == directly on arrays inside if statement
if np.allclose(DotAB_C, DotA_BC):
  print("Matrix has Associative Property.")
else:
  print("Failed!")
  
# 2. Commutative Property AB != BA
print(f"\nChecking Commutative Property of \n {DotAB} and \n {DotBA}")

if np.allclose(DotAB, DotBA):
  print("Failed!")
else:
  print("Matrix has Commutative Property.")
  
# 3. Distributive Property: A(B + C) = AB + AC  ;  (B + C)A = BA + CA
# A(B + C) = AB + AC
DotAAddBC = np.dot(A, AddBC)
AddDotABDotAC = DotAB + DotAC
print(f"\nChecking Distributive Property for [A(B + C) = AB + AC] of \n {DotAAddBC} and \n {AddDotABDotAC}")
if np.allclose(DotAAddBC, AddDotABDotAC):
  print("Matrix has Distributive Property for [A(B + C) = AB + AC].")
else:
  print("Failed!")
  
# (B + C)A = BA + CA
AddBCDotA = np.dot(AddBC, A)
AddDotBADotCA = DotBA + DotCA
print(f"Checking Distributive Property for [(B + C)A = BA + CA] of \n {AddBCDotA} and \n {AddDotBADotCA}")
if np.allclose(AddBCDotA, AddDotBADotCA):
  print("Matrix has Distributive Property for [(B + C)A = BA + CA].")
else:
  print("Failed!")
  
# 4. Multiplicative identity: IA = AI = A
print(f"Checking Multiplicative Identity of \n {DotIA} and \n {DotAI} and \n {A}")
if np.allclose(DotIA, DotAI, A):
  print("Matrix has Multiplicative Identity Property.")
else:
  print("Failed!")
  
# 5. Multiplicative zero: 0A = A0 = 0
print(f"Checking Multiplicative Zero Property of \n {DotZA} and \n {DotAZ} and \n {Z}")
if np.allclose(DotZA, DotAZ, Z):
  print("Matrix has Multiplicative Zero Property.")
else:
  print("Failed!")