import random
import time
import matplotlib.pyplot as plt
import numpy as np
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr
def plot_basic(data_sizes, timings, title):
    """Функция для построения графика зависимости времени от размера массива без регрессии"""
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, timings, marker='o', label=title)
    plt.title(f'Время сортировки selection_sort: {title} без регрессии')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
def plot_regression_curve(data_sizes, timings, title):
    """Функция для построения графика с регрессионной кривой"""
    plt.figure(figsize=(10, 6))
    # Полиномиальная регрессия (3-го порядка)
    coefficients = np.polyfit(data_sizes, timings, 3)  # Степень полинома можно менять (например, 2 или 3)
    polynomial = np.poly1d(coefficients)
    # Плавная кривая для отображения
    x_smooth = np.linspace(min(data_sizes), max(data_sizes), 500)
    y_smooth = polynomial(x_smooth)
    plt.plot(x_smooth, y_smooth, color='r', linestyle='--', label='Регрессионная кривая')
    plt.title(f'Регрессионная кривая selection_sort: {title}')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
data_sizes = [10000, 15000, 20000, 25000, 30000]
results = {
    "Отсортированный": [],
    "90% отсортирован": [],
    "Убывающий": [],
    "Случайный": []
}
# Запуск сортировки и запись времени
for size in data_sizes:
    # Полный массив
    full_data = [random.randint(0, 100000) for _ in range(size)]
    full_data.sort()
    start_time = time.time()
    selection_sort(full_data)
    end_time = time.time()
    results["Отсортированный"].append(end_time - start_time)
    # Массив с 90% отсортированными элементами
    sub_data = full_data[:int(size * 0.9)]
    sub_data.sort()  # Сортируем 90%
    sub_data += [random.randint(0, 100000) for _ in range(int(size * 0.1))]  # Добавляем 10% неотсортированных
    start_time = time.time()
    selection_sort(sub_data)
    end_time = time.time()
    results["90% отсортирован"].append(end_time - start_time)
    # Массив, отсортированный по убыванию
    descending_data = sorted(full_data, reverse=True)
    start_time = time.time()
    selection_sort(descending_data)
    end_time = time.time()
    results["Убывающий"].append(end_time - start_time)
    # Случайный массив
    random_data = [random.randint(0, 100000) for _ in range(size)]
    start_time = time.time()
    selection_sort(random_data)
    end_time = time.time()
    results["Случайный"].append(end_time - start_time)
# Сначала строим графики без регрессии
plt.figure(figsize=(10, 6))
for category, timings in results.items():
    plt.plot(data_sizes, timings, marker='o', label=category)
plt.title('selection_sort: Время сортировки в зависимости от размера массива')
plt.xlabel('Количество элементов')
plt.ylabel('Время сортировки (секунды)')
plt.legend()
plt.grid()
plt.show(block = False)
for category, timings in results.items():
    for size, timing in zip(data_sizes, timings):
        print(f"Время сортировки {category} массива: {timing:.6f} секунд для {size} элементов")
    plot_basic(data_sizes, timings, category)
# Теперь строим отдельные графики с регрессионной кривой
for category, timings in results.items():
    plot_regression_curve(data_sizes, timings, category)
plt.show()
def insertion_sort(arr):
    """Сортировка является устойчивой из-за сдвига"""
    n = len(arr)
    for i in range(1, n):
        x, j = arr[i], i
        while j > 0 and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1
        arr[j] = x
    return arr
def plot_basic(data_sizes, timings, title):
    """Функция для построения графика зависимости времени от размера массива без регрессии"""
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, timings, marker='o', label=title)
    plt.title(f'Время сортировки insertion_sort: {title} без регрессии')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
def plot_regression_curve(data_sizes, timings, title):
    """Функция для построения графика с регрессионной кривой"""
    plt.figure(figsize=(10, 6))
    # Полиномиальная регрессия (3-го порядка)
    coefficients = np.polyfit(data_sizes, timings, 3)  # Степень полинома можно менять (например, 2 или 3)
    polynomial = np.poly1d(coefficients)
    # Плавная кривая для отображения
    x_smooth = np.linspace(min(data_sizes), max(data_sizes), 500)
    y_smooth = polynomial(x_smooth)
    plt.plot(x_smooth, y_smooth, color='r', linestyle='--', label='Регрессионная кривая')
    plt.title(f'Регрессионная кривая insertion_sort: {title}')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
data_sizes = [10000, 15000, 20000, 25000, 30000]
results = {
    "Отсортированный": [],
    "90% отсортирован": [],
    "Убывающий": [],
    "Случайный": []
}
# Запуск сортировки и запись времени
for size in data_sizes:
    # Полный массив
    full_data = [random.randint(0, 100000) for _ in range(size)]
    full_data.sort()
    start_time = time.time()
    insertion_sort(full_data)
    end_time = time.time()
    results["Отсортированный"].append(end_time - start_time)
    # Массив с 90% отсортированными элементами
    sub_data = full_data[:int(size * 0.9)]
    sub_data.sort()  # Сортируем 90%
    sub_data += [random.randint(0, 100000) for _ in range(int(size * 0.1))]  # Добавляем 10% неотсортированных
    start_time = time.time()
    insertion_sort(sub_data)
    end_time = time.time()
    results["90% отсортирован"].append(end_time - start_time)
    # Массив, отсортированный по убыванию
    descending_data = sorted(full_data, reverse=True)
    start_time = time.time()
    insertion_sort(descending_data)
    end_time = time.time()
    results["Убывающий"].append(end_time - start_time)
    # Случайный массив
    random_data = [random.randint(0, 100000) for _ in range(size)]
    start_time = time.time()
    insertion_sort(random_data)
    end_time = time.time()
    results["Случайный"].append(end_time - start_time)
# Сначала строим графики без регрессии
plt.figure(figsize=(10, 6))
for category, timings in results.items():
    plt.plot(data_sizes, timings, marker='o', label=category)
plt.title('insertion_sort: Время сортировки в зависимости от размера массива')
plt.xlabel('Количество элементов')
plt.ylabel('Время сортировки (секунды)')
plt.legend()
plt.grid()
plt.show(block = False)
for category, timings in results.items():
    for size, timing in zip(data_sizes, timings):
        print(f"Время сортировки {category} массива: {timing:.6f} секунд для {size} элементов")
    plot_basic(data_sizes, timings, category)
# Теперь строим отдельные графики с регрессионной кривой
for category, timings in results.items():
    plot_regression_curve(data_sizes, timings, category)
plt.show()
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
def plot_basic(data_sizes, timings, title):
    """Функция для построения графика зависимости времени от размера массива без регрессии"""
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, timings, marker='o', label=title)
    plt.title(f'Время сортировки bubble_sort: {title} без регрессии')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
def plot_regression_curve(data_sizes, timings, title):
    """Функция для построения графика с регрессионной кривой"""
    plt.figure(figsize=(10, 6))
    # Полиномиальная регрессия (3-го порядка)
    coefficients = np.polyfit(data_sizes, timings, 3)  # Степень полинома можно менять (например, 2 или 3)
    polynomial = np.poly1d(coefficients)
    # Плавная кривая для отображения
    x_smooth = np.linspace(min(data_sizes), max(data_sizes), 500)
    y_smooth = polynomial(x_smooth)
    plt.plot(x_smooth, y_smooth, color='r', linestyle='--', label='Регрессионная кривая')
    plt.title(f'Регрессионная кривая bubble_sort: {title}')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
data_sizes = [10000, 15000, 20000, 25000, 30000]
results = {
    "Отсортированный": [],
    "90% отсортирован": [],
    "Убывающий": [],
    "Случайный": []
}
# Запуск сортировки и запись времени
for size in data_sizes:
    # Полный массив
    full_data = [random.randint(0, 100000) for _ in range(size)]
    full_data.sort()
    start_time = time.time()
    bubble_sort(full_data)
    end_time = time.time()
    results["Отсортированный"].append(end_time - start_time)
    # Массив с 90% отсортированными элементами
    sub_data = full_data[:int(size * 0.9)]
    sub_data.sort()  # Сортируем 90%
    sub_data += [random.randint(0, 100000) for _ in range(int(size * 0.1))]  # Добавляем 10% неотсортированных
    start_time = time.time()
    bubble_sort(sub_data)
    end_time = time.time()
    results["90% отсортирован"].append(end_time - start_time)
    # Массив, отсортированный по убыванию
    descending_data = sorted(full_data, reverse=True)
    start_time = time.time()
    bubble_sort(descending_data)
    end_time = time.time()
    results["Убывающий"].append(end_time - start_time)
    # Случайный массив
    random_data = [random.randint(0, 100000) for _ in range(size)]
    start_time = time.time()
    bubble_sort(random_data)
    end_time = time.time()
    results["Случайный"].append(end_time - start_time)
# Сначала строим графики без регрессии
plt.figure(figsize=(10, 6))
for category, timings in results.items():
    plt.plot(data_sizes, timings, marker='o', label=category)
plt.title('bubble_sort: Время сортировки в зависимости от размера массива')
plt.xlabel('Количество элементов')
plt.ylabel('Время сортировки (секунды)')
plt.legend()
plt.grid()
plt.show(block = False)
for category, timings in results.items():
    for size, timing in zip(data_sizes, timings):
        print(f"Время сортировки {category} массива: {timing:.6f} секунд для {size} элементов")
    plot_basic(data_sizes, timings, category)
# Теперь строим отдельные графики с регрессионной кривой
for category, timings in results.items():
    plot_regression_curve(data_sizes, timings, category)
plt.show()
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        merge_sort(left_half)
        merge_sort(right_half)
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
    return arr
def plot_basic(data_sizes, timings, title):
    """Функция для построения графика зависимости времени от размера массива без регрессии"""
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, timings, marker='o', label=title)
    plt.title(f'Время сортировки merge_sort: {title} без регрессии')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
def plot_regression_curve(data_sizes, timings, title):
    """Функция для построения графика с регрессионной кривой"""
    plt.figure(figsize=(10, 6))
    # Полиномиальная регрессия (3-го порядка)
    coefficients = np.polyfit(data_sizes, timings, 3)  # Степень полинома можно менять (например, 2 или 3)
    polynomial = np.poly1d(coefficients)
    # Плавная кривая для отображения
    x_smooth = np.linspace(min(data_sizes), max(data_sizes), 500)
    y_smooth = polynomial(x_smooth)
    plt.plot(x_smooth, y_smooth, color='r', linestyle='--', label='Регрессионная кривая')
    plt.title(f'Регрессионная кривая merge_sort: {title}')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
data_sizes = [100000, 150000, 200000, 250000, 300000]
results = {
    "Отсортированный": [],
    "90% отсортирован": [],
    "Убывающий": [],
    "Случайный": []
}
# Запуск сортировки и запись времени
for size in data_sizes:
    # Полный массив
    full_data = [random.randint(0, 1000000) for _ in range(size)]
    full_data.sort()
    start_time = time.time()
    merge_sort(full_data)
    end_time = time.time()
    results["Отсортированный"].append(end_time - start_time)
    # Массив с 90% отсортированными элементами
    sub_data = full_data[:int(size * 0.9)]
    sub_data.sort()  # Сортируем 90%
    sub_data += [random.randint(0, 1000000) for _ in range(int(size * 0.1))]  # Добавляем 10% неотсортированных
    start_time = time.time()
    merge_sort(sub_data)
    end_time = time.time()
    results["90% отсортирован"].append(end_time - start_time)
    # Массив, отсортированный по убыванию
    descending_data = sorted(full_data, reverse=True)
    start_time = time.time()
    merge_sort(descending_data)
    end_time = time.time()
    results["Убывающий"].append(end_time - start_time)
    # Случайный массив
    random_data = [random.randint(0, 1000000) for _ in range(size)]
    start_time = time.time()
    merge_sort(random_data)
    end_time = time.time()
    results["Случайный"].append(end_time - start_time)
# Сначала строим графики без регрессии
plt.figure(figsize=(10, 6))
for category, timings in results.items():
    plt.plot(data_sizes, timings, marker='o', label=category)
plt.title('merge_sort: Время сортировки в зависимости от размера массива')
plt.xlabel('Количество элементов')
plt.ylabel('Время сортировки (секунды)')
plt.legend()
plt.grid()
plt.show(block = False)
for category, timings in results.items():
    for size, timing in zip(data_sizes, timings):
        print(f"Время сортировки {category} массива: {timing:.6f} секунд для {size} элементов")
    plot_basic(data_sizes, timings, category)
# Теперь строим отдельные графики с регрессионной кривой
for category, timings in results.items():
    plot_regression_curve(data_sizes, timings, category)
plt.show()
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr
def plot_basic(data_sizes, timings, title):
    """Функция для построения графика зависимости времени от размера массива без регрессии"""
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, timings, marker='o', label=title)
    plt.title(f'Время сортировки shell_sort: {title} без регрессии')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
def plot_regression_curve(data_sizes, timings, title):
    """Функция для построения графика с регрессионной кривой"""
    plt.figure(figsize=(10, 6))
    # Полиномиальная регрессия (3-го порядка)
    coefficients = np.polyfit(data_sizes, timings, 3)  # Степень полинома можно менять (например, 2 или 3)
    polynomial = np.poly1d(coefficients)
    # Плавная кривая для отображения
    x_smooth = np.linspace(min(data_sizes), max(data_sizes), 500)
    y_smooth = polynomial(x_smooth)
    plt.plot(x_smooth, y_smooth, color='r', linestyle='--', label='Регрессионная кривая')
    plt.title(f'Регрессионная кривая shell_sort: {title}')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
data_sizes = [100000, 150000, 200000, 250000, 300000]
results = {
    "Отсортированный": [],
    "90% отсортирован": [],
    "Убывающий": [],
    "Случайный": []
}
# Запуск сортировки и запись времени
for size in data_sizes:
    # Полный массив
    full_data = [random.randint(0, 1000000) for _ in range(size)]
    full_data.sort()
    start_time = time.time()
    shell_sort(full_data)
    end_time = time.time()
    results["Отсортированный"].append(end_time - start_time)
    # Массив с 90% отсортированными элементами
    sub_data = full_data[:int(size * 0.9)]
    sub_data.sort()  # Сортируем 90%
    sub_data += [random.randint(0, 1000000) for _ in range(int(size * 0.1))]  # Добавляем 10% неотсортированных
    start_time = time.time()
    shell_sort(sub_data)
    end_time = time.time()
    results["90% отсортирован"].append(end_time - start_time)
    # Массив, отсортированный по убыванию
    descending_data = sorted(full_data, reverse=True)
    start_time = time.time()
    shell_sort(descending_data)
    end_time = time.time()
    results["Убывающий"].append(end_time - start_time)
    # Случайный массив
    random_data = [random.randint(0, 1000000) for _ in range(size)]
    start_time = time.time()
    shell_sort(random_data)
    end_time = time.time()
    results["Случайный"].append(end_time - start_time)
# Сначала строим графики без регрессии
plt.figure(figsize=(10, 6))
for category, timings in results.items():
    plt.plot(data_sizes, timings, marker='o', label=category)
plt.title('shell_sort: Время сортировки в зависимости от размера массива')
plt.xlabel('Количество элементов')
plt.ylabel('Время сортировки (секунды)')
plt.legend()
plt.grid()
plt.show(block = False)
for category, timings in results.items():
    for size, timing in zip(data_sizes, timings):
        print(f"Время сортировки {category} массива: {timing:.6f} секунд для {size} элементов")
    plot_basic(data_sizes, timings, category)
# Теперь строим отдельные графики с регрессионной кривой
for category, timings in results.items():
    plot_regression_curve(data_sizes, timings, category)
plt.show()
def hibbard_sort(arr):
    n = len(arr)
    gap = 1
    while gap < n:
        gap = 2 * gap + 1
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap = (gap - 1) // 2
    return arr
def plot_basic(data_sizes, timings, title):
    """Функция для построения графика зависимости времени от размера массива без регрессии"""
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, timings, marker='o', label=title)
    plt.title(f'Время сортировки hibbard_sort: {title} без регрессии')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
def plot_regression_curve(data_sizes, timings, title):
    """Функция для построения графика с регрессионной кривой"""
    plt.figure(figsize=(10, 6))
    # Полиномиальная регрессия (3-го порядка)
    coefficients = np.polyfit(data_sizes, timings, 3)  # Степень полинома можно менять (например, 2 или 3)
    polynomial = np.poly1d(coefficients)
    # Плавная кривая для отображения
    x_smooth = np.linspace(min(data_sizes), max(data_sizes), 500)
    y_smooth = polynomial(x_smooth)
    plt.plot(x_smooth, y_smooth, color='r', linestyle='--', label='Регрессионная кривая')
    plt.title(f'Регрессионная кривая hibbard_sort: {title}')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
data_sizes = [100000, 150000, 200000, 250000, 300000]
results = {
    "Отсортированный": [],
    "90% отсортирован": [],
    "Убывающий": [],
    "Случайный": []
}
# Запуск сортировки и запись времени
for size in data_sizes:
    # Полный массив
    full_data = [random.randint(0, 1000000) for _ in range(size)]
    full_data.sort()
    start_time = time.time()
    hibbard_sort(full_data)
    end_time = time.time()
    results["Отсортированный"].append(end_time - start_time)
    # Массив с 90% отсортированными элементами
    sub_data = full_data[:int(size * 0.9)]
    sub_data.sort()  # Сортируем 90%
    sub_data += [random.randint(0, 1000000) for _ in range(int(size * 0.1))]  # Добавляем 10% неотсортированных
    start_time = time.time()
    hibbard_sort(sub_data)
    end_time = time.time()
    results["90% отсортирован"].append(end_time - start_time)
    # Массив, отсортированный по убыванию
    descending_data = sorted(full_data, reverse=True)
    start_time = time.time()
    hibbard_sort(descending_data)
    end_time = time.time()
    results["Убывающий"].append(end_time - start_time)
    # Случайный массив
    random_data = [random.randint(0, 1000000) for _ in range(size)]
    start_time = time.time()
    hibbard_sort(random_data)
    end_time = time.time()
    results["Случайный"].append(end_time - start_time)
# Сначала строим графики без регрессии
plt.figure(figsize=(10, 6))
for category, timings in results.items():
    plt.plot(data_sizes, timings, marker='o', label=category)
plt.title('hibbard_sort: Время сортировки в зависимости от размера массива')
plt.xlabel('Количество элементов')
plt.ylabel('Время сортировки (секунды)')
plt.legend()
plt.grid()
plt.show(block = False)
for category, timings in results.items():
    for size, timing in zip(data_sizes, timings):
        print(f"Время сортировки {category} массива: {timing:.6f} секунд для {size} элементов")
    plot_basic(data_sizes, timings, category)
# Теперь строим отдельные графики с регрессионной кривой
for category, timings in results.items():
    plot_regression_curve(data_sizes, timings, category)
plt.show()
def pratt_sort(arr):
    n = len(arr)
    gaps = []
    i, j = 0, 0
    while True:
        gap = (2 ** i) * (3 ** j)
        if gap > n:
            if i == 0: break
            j += 1
            i = 0
        else:
            gaps.append(gap)
            i += 1
    gaps.sort(reverse=True)
    for gap in gaps:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
    return arr
def plot_basic(data_sizes, timings, title):
    """Функция для построения графика зависимости времени от размера массива без регрессии"""
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, timings, marker='o', label=title)
    plt.title(f'Время сортировки pratt_sort: {title} без регрессии')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
def plot_regression_curve(data_sizes, timings, title):
    """Функция для построения графика с регрессионной кривой"""
    plt.figure(figsize=(10, 6))
    # Полиномиальная регрессия (3-го порядка)
    coefficients = np.polyfit(data_sizes, timings, 3)  # Степень полинома можно менять (например, 2 или 3)
    polynomial = np.poly1d(coefficients)
    # Плавная кривая для отображения
    x_smooth = np.linspace(min(data_sizes), max(data_sizes), 500)
    y_smooth = polynomial(x_smooth)
    plt.plot(x_smooth, y_smooth, color='r', linestyle='--', label='Регрессионная кривая')
    plt.title(f'Регрессионная кривая pratt_sort: {title}')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
data_sizes = [100000, 150000, 200000, 250000, 300000]
results = {
    "Отсортированный": [],
    "90% отсортирован": [],
    "Убывающий": [],
    "Случайный": []
}
# Запуск сортировки и запись времени
for size in data_sizes:
    # Полный массив
    full_data = [random.randint(0, 1000000) for _ in range(size)]
    full_data.sort()
    start_time = time.time()
    pratt_sort(full_data)
    end_time = time.time()
    results["Отсортированный"].append(end_time - start_time)
    # Массив с 90% отсортированными элементами
    sub_data = full_data[:int(size * 0.9)]
    sub_data.sort()  # Сортируем 90%
    sub_data += [random.randint(0, 1000000) for _ in range(int(size * 0.1))]  # Добавляем 10% неотсортированных
    start_time = time.time()
    pratt_sort(sub_data)
    end_time = time.time()
    results["90% отсортирован"].append(end_time - start_time)
    # Массив, отсортированный по убыванию
    descending_data = sorted(full_data, reverse=True)
    start_time = time.time()
    pratt_sort(descending_data)
    end_time = time.time()
    results["Убывающий"].append(end_time - start_time)
    # Случайный массив
    random_data = [random.randint(0, 1000000) for _ in range(size)]
    start_time = time.time()
    pratt_sort(random_data)
    end_time = time.time()
    results["Случайный"].append(end_time - start_time)
# Сначала строим графики без регрессии
plt.figure(figsize=(10, 6))
for category, timings in results.items():
    plt.plot(data_sizes, timings, marker='o', label=category)
plt.title('pratt_sort: Время сортировки в зависимости от размера массива')
plt.xlabel('Количество элементов')
plt.ylabel('Время сортировки (секунды)')
plt.legend()
plt.grid()
plt.show(block = False)
for category, timings in results.items():
    for size, timing in zip(data_sizes, timings):
        print(f"Время сортировки {category} массива: {timing:.6f} секунд для {size} элементов")
    plot_basic(data_sizes, timings, category)
# Теперь строим отдельные графики с регрессионной кривой
for category, timings in results.items():
    plot_regression_curve(data_sizes, timings, category)

plt.show()
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
def plot_basic(data_sizes, timings, title):
    """Функция для построения графика зависимости времени от размера массива без регрессии"""
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, timings, marker='o', label=title)
    plt.title(f'Время сортировки quick_sort: {title} без регрессии')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
def plot_regression_curve(data_sizes, timings, title):
    """Функция для построения графика с регрессионной кривой"""
    plt.figure(figsize=(10, 6))
    # Полиномиальная регрессия (3-го порядка)
    coefficients = np.polyfit(data_sizes, timings, 3)  # Степень полинома можно менять (например, 2 или 3)
    polynomial = np.poly1d(coefficients)
    # Плавная кривая для отображения
    x_smooth = np.linspace(min(data_sizes), max(data_sizes), 500)
    y_smooth = polynomial(x_smooth)
    plt.plot(x_smooth, y_smooth, color='r', linestyle='--', label='Регрессионная кривая')
    plt.title(f'Регрессионная кривая quick_sort: {title}')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
data_sizes = [100000, 150000, 200000, 250000, 300000]
results = {
    "Отсортированный": [],
    "90% отсортирован": [],
    "Убывающий": [],
    "Случайный": []
}
# Запуск сортировки и запись времени
for size in data_sizes:
    # Полный массив
    full_data = [random.randint(0, 1000000) for _ in range(size)]
    full_data.sort()
    start_time = time.time()
    quick_sort(full_data)
    end_time = time.time()
    results["Отсортированный"].append(end_time - start_time)
    # Массив с 90% отсортированными элементами
    sub_data = full_data[:int(size * 0.9)]
    sub_data.sort()  # Сортируем 90%
    sub_data += [random.randint(0, 1000000) for _ in range(int(size * 0.1))]  # Добавляем 10% неотсортированных
    start_time = time.time()
    quick_sort(sub_data)
    end_time = time.time()
    results["90% отсортирован"].append(end_time - start_time)
    # Массив, отсортированный по убыванию
    descending_data = sorted(full_data, reverse=True)
    start_time = time.time()
    quick_sort(descending_data)
    end_time = time.time()
    results["Убывающий"].append(end_time - start_time)
    # Случайный массив
    random_data = [random.randint(0, 1000000) for _ in range(size)]
    start_time = time.time()
    quick_sort(random_data)
    end_time = time.time()
    results["Случайный"].append(end_time - start_time)
# Сначала строим графики без регрессии
plt.figure(figsize=(10, 6))
for category, timings in results.items():
    plt.plot(data_sizes, timings, marker='o', label=category)
plt.title('quick_sort: Время сортировки в зависимости от размера массива')
plt.xlabel('Количество элементов')
plt.ylabel('Время сортировки (секунды)')
plt.legend()
plt.grid()
plt.show(block = False)
for category, timings in results.items():
    for size, timing in zip(data_sizes, timings):
        print(f"Время сортировки {category} массива: {timing:.6f} секунд для {size} элементов")
    plot_basic(data_sizes, timings, category)
# Теперь строим отдельные графики с регрессионной кривой
for category, timings in results.items():
    plot_regression_curve(data_sizes, timings, category)
plt.show()
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
def plot_basic(data_sizes, timings, title):
    """Функция для построения графика зависимости времени от размера массива без регрессии"""
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, timings, marker='o', label=title)
    plt.title(f'Время сортировки heap_sort: {title} без регрессии')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
def plot_regression_curve(data_sizes, timings, title):
    """Функция для построения графика с регрессионной кривой"""
    plt.figure(figsize=(10, 6))
    # Полиномиальная регрессия (3-го порядка)
    coefficients = np.polyfit(data_sizes, timings, 3)  # Степень полинома можно менять (например, 2 или 3)
    polynomial = np.poly1d(coefficients)
    # Плавная кривая для отображения
    x_smooth = np.linspace(min(data_sizes), max(data_sizes), 500)
    y_smooth = polynomial(x_smooth)
    plt.plot(x_smooth, y_smooth, color='r', linestyle='--', label='Регрессионная кривая')
    plt.title(f'Регрессионная кривая heap_sort: {title}')
    plt.xlabel('Количество элементов')
    plt.ylabel('Время сортировки (секунды)')
    plt.legend()
    plt.grid()
    plt.show(block=False)
data_sizes = [100000, 150000, 200000, 250000, 300000]
results = {
    "Отсортированный": [],
    "90% отсортирован": [],
    "Убывающий": [],
    "Случайный": []
}
# Запуск сортировки и запись времени
for size in data_sizes:
    # Полный массив
    full_data = [random.randint(0, 1000000) for _ in range(size)]
    full_data.sort()
    start_time = time.time()
    heap_sort(full_data)
    end_time = time.time()
    results["Отсортированный"].append(end_time - start_time)
    # Массив с 90% отсортированными элементами
    sub_data = full_data[:int(size * 0.9)]
    sub_data.sort()  # Сортируем 90%
    sub_data += [random.randint(0, 1000000) for _ in range(int(size * 0.1))]  # Добавляем 10% неотсортированных
    start_time = time.time()
    heap_sort(sub_data)
    end_time = time.time()
    results["90% отсортирован"].append(end_time - start_time)
    # Массив, отсортированный по убыванию
    descending_data = sorted(full_data, reverse=True)
    start_time = time.time()
    heap_sort(descending_data)
    end_time = time.time()
    results["Убывающий"].append(end_time - start_time)
    # Случайный массив
    random_data = [random.randint(0, 1000000) for _ in range(size)]
    start_time = time.time()
    heap_sort(random_data)
    end_time = time.time()
    results["Случайный"].append(end_time - start_time)
# Сначала строим графики без регрессии
plt.figure(figsize=(10, 6))
for category, timings in results.items():
    plt.plot(data_sizes, timings, marker='o', label=category)
plt.title('heap_sort: Время сортировки в зависимости от размера массива')
plt.xlabel('Количество элементов')
plt.ylabel('Время сортировки (секунды)')
plt.legend()
plt.grid()
plt.show(block = False)
for category, timings in results.items():
    for size, timing in zip(data_sizes, timings):
        print(f"Время сортировки {category} массива: {timing:.6f} секунд для {size} элементов")
    plot_basic(data_sizes, timings, category)
# Теперь строим отдельные графики с регрессионной кривой
for category, timings in results.items():
    plot_regression_curve(data_sizes, timings, category)
plt.show()
# Функция для замера времени сортировки
def measure_time(sort_func, sizes):
    timings = []
    for size in sizes:
        arr = [random.randint(0, 1000000) for _ in range(size)]
        start_time = time.time()
        sort_func(arr)
        timings.append(time.time() - start_time)
    return timings
# Построение графика
def plot_results(data_sizes, timings_dict):
    plt.figure(figsize=(12, 8))
    for label, timings in timings_dict.items():
        plt.plot(data_sizes, timings, marker='o', label=label)
    plt.xlabel('Количество элементов')
    plt.ylabel('Время выполнения (секунды)')
    plt.title('Графики с квадратичной асимптотикой')
    plt.legend()
    plt.grid(True)
    plt.show()
# Основной код
data_sizes = [10000, 15000, 20000, 25000, 30000]
timings_dict = {
    'selection Sort': measure_time(selection_sort, data_sizes),
    'insertion Sort': measure_time(insertion_sort, data_sizes),
    'bubble Sort': measure_time(bubble_sort, data_sizes),

}
# Построение графиков
plot_results(data_sizes, timings_dict)
# Построение графика
def plot_results(data_sizes, timings_dict):
    plt.figure(figsize=(12, 8))
    for label, timings in timings_dict.items():
        plt.plot(data_sizes, timings, marker='o', label=label)
    plt.xlabel('Количество элементов')
    plt.ylabel('Время выполнения (секунды)')
    plt.title('Графики с не квадратичной асимптотикой')
    plt.legend()
    plt.grid(True)
    plt.show()
# Основной код
data_sizes = [100000, 150000, 200000, 250000, 300000]
timings_dict = {
    'Merge Sort': measure_time(merge_sort, data_sizes),
    'Shell Sort': measure_time(shell_sort, data_sizes),
    'Hibbard Sort': measure_time(hibbard_sort, data_sizes),
    'Pratt Sort': measure_time(pratt_sort, data_sizes),
    'Quick Sort': measure_time(quick_sort, data_sizes),
    'Heap Sort': measure_time(heap_sort, data_sizes)
}
# Построение графиков
plot_results(data_sizes, timings_dict)
