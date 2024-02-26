import matplotlib.pyplot as plt

def solve_linear_system(A, b):
    
    # Rozwiązanie układu A * x = b
    # ZAŁOŻENIE: A to macież 3x3, a b to wektor 3x1.
    
    # Wyznacznik A
    det_A = (A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) -
             A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) +
             A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]))

    if det_A == 0:
        raise ValueError("Układ nie ma jednego rozwiązania")

    # Macierz odwrotna do A
    inv_A = [[(A[1][1]*A[2][2] - A[1][2]*A[2][1])/det_A, -(A[0][1]*A[2][2] - A[0][2]*A[2][1])/det_A, (A[0][1]*A[1][2] - A[0][2]*A[1][1])/det_A],
             [-(A[1][0]*A[2][2] - A[1][2]*A[2][0])/det_A, (A[0][0]*A[2][2] - A[0][2]*A[2][0])/det_A, -(A[0][0]*A[1][2] - A[0][2]*A[1][0])/det_A],
             [(A[1][0]*A[2][1] - A[1][1]*A[2][0])/det_A, -(A[0][0]*A[2][1] - A[0][1]*A[2][0])/det_A, (A[0][0]*A[1][1] - A[0][1]*A[1][0])/det_A]]

    # x = inv_A * b
    x = [0, 0, 0]
    for i in range(3):
        x[i] = sum(inv_A[i][j] * b[j] for j in range(3))

    return x

def least_squares_quadratic_fit(x_data, y_data):

    # Budowanie macierzy dla MNK
    A = [[sum(x**4 for x in x_data), sum(x**3 for x in x_data), sum(x**2 for x in x_data)],
         [sum(x**3 for x in x_data), sum(x**2 for x in x_data), sum(x for x in x_data)],
         [sum(x**2 for x in x_data), sum(x for x in x_data), len(x_data)]]
    b = [sum(x_data[i]**2 * y_data[i] for i in range(len(x_data))),
         sum(x_data[i] * y_data[i] for i in range(len(x_data))),
         sum(y_data)]

    # Rozwiązywanie układu w celu znalezienia współczynników
    a, b, c = solve_linear_system(A, b)
    return a, b, c

# Wczytywanie danych z pliku
def read_data_from_file(file_path):
    x_data = []
    y_data = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = line.strip().split(',')
            x_data.append(float(x))
            y_data.append(float(y))
    return x_data, y_data

x_data, y_data = read_data_from_file('test.txt')

# Aproksymacja
a, b, c = least_squares_quadratic_fit(x_data, y_data)
x_min = min(x_data)
x_max = max(x_data)
x_interp_step = (x_max - x_min) / 100  # For example, divide the range into 100 steps
x_interp = [x_min + i * x_interp_step for i in range(101)]  # Generating 101 points for smooth curve

y_fit = [a*x**2 + b*x + c for x in x_interp]

# Wykres
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='red', label='Punkty danych')
plt.plot(x_interp, y_fit, label='Dopasowanie kwadratowe (MNK)')
plt.title('Dopasowanie kwadratowe metodą najmniejszych kwadratów')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
