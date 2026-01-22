import pandas as pd

# 1. Загружаем файл (используем r перед кавычками, чтобы Windows-путь сработал)
df = pd.read_csv(r"C:\Users\esimo\Desktop\nyc_taxi_project\dags\data\taxi_data.csv")

# 2. Фильтруем данные
# Ищем поездки, где дистанция примерно 5 миль (от 4.9 до 5.1)
filtered_df = df[ (df['trip_distance'] >= 4.9) & (df['trip_distance'] <= 5.1) ]

# 3. Берем первые 10 таких поездок
first_10 = filtered_df.head(100)

# 4. Выводим цену каждой из этих 10 поездок
print("--- Цены за первые 10 поездок (~5 миль) ---")
# iterrows() позволяет пройтись по строкам, если нужно красивое отображение
for index, row in first_10.iterrows():
    print(f"Поездка #{index}: Дистанция {row['trip_distance']} миль -> Цена ${row['total_amount']}")

# 5. Считаем среднее значение цены только для этих 10 поездок
average_price = first_10['total_amount'].mean()

print("\n" + "="*30)
print(f"Средняя цена для этих 10 поездок: ${average_price:.2f}")