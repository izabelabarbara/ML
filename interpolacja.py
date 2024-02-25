import matplotlib.pyplot as plt

def lagrange_interpolation(x_points, y_points, x):
    total = 0
    n = len(x_points)
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if j != i:
                term = term * (x - x_points[j]) / (x_points[i] - x_points[j])
        total += term
    return total

x_data = []
y_data = []

# Wczytywanie danych z pliku
with open('test.txt', 'r') as file:
    for line in file:
        x, y = line.strip().split(',')
        x_data.append(float(x))
        y_data.append(float(y))

# Interpolacja
x_interp = [min(x_data) + i * (max(x_data) - min(x_data)) / 99 for i in range(100)]
y_interp = [lagrange_interpolation(x_data, y_data, xi) for xi in x_interp]

# Wykres
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='red', label='Punkty danych')
plt.plot(x_interp, y_interp, label='Wielomian interpolacyjny Lagrange’a')
plt.title('Interpolacja Lagrange’a')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()