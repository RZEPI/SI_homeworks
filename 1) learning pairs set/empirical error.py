import numpy as np

def generate_matrix(learn_pairs):
    matrix = []
    wector = []
    for pair in learn_pairs:
        row = []
        row.append(pair[0]*pair[0])
        row.append(pair[0])
        matrix.append(row)
        wector.append([pair[1]])
    matrix = np.array(matrix)
    wector = np.array(wector)
    return matrix, wector


x_d_pairs =[(-1.21, 2.3), (0.10, 3.4), (0.40, 5.1), (0.82, 6.3)]
A, d = generate_matrix(x_d_pairs)
w = np.linalg.inv(np.transpose(A).dot(A)).dot(np.transpose(A)).dot(d)
E_emp = np.transpose(d-A.dot(w)).dot(d-A.dot(w))
E_emp = list(E_emp)
E_emp = E_emp[0][0]
print("E(w) = " + str(E_emp))

w = list(w)

print("f(x) = {0}*x^2 + {1}".format(w[0][0], w[1][0]))