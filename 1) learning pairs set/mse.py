import numpy as np

def generate_matrix(learn_pairs):
    matrix = []
    wector = []
    for pair in learn_pairs:
        row = []
        row.append(pair[0])
        row.append(pair[1])
        row.append(pair[0]*pair[1])
        row.append(pair[0]/pair[1])
        row.append(pair[0]*pair[0])
        row.append(pair[1]*pair[1])
        matrix.append(row)
        wector.append([pair[2]])
    matrix = np.array(matrix)
    wector = np.array(wector)
    return matrix, wector


learn_pairs = [(0.2, 0.3, 0.8), (-0.3, 0.4, 0.2), (-0.5, 3.3, -0.3), (-0.1, 4.8, 1.2), (-1.0, 3.2, 1.6), (-0.3, 7.2, 0.5), (0.1, 3.4, -0.2)]
A, d = generate_matrix(learn_pairs)
w = np.linalg.inv(np.transpose(A).dot(A)).dot(np.transpose(A)).dot(d)
E_emp = np.transpose(d-A.dot(w)).dot(d-A.dot(w))
E_emp = list(E_emp)
E_emp = E_emp[0][0]
print("E(w) = " + str(E_emp))

A = np.delete(A, [4,5], 1)

w = np.linalg.inv(np.transpose(A).dot(A)).dot(np.transpose(A)).dot(d)
E_emp = np.transpose(d-A.dot(w)).dot(d-A.dot(w))
E_emp = list(E_emp)
E_emp = E_emp[0][0]
print("E(w) = " + str(E_emp))