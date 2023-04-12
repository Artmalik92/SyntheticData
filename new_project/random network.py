import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from scipy.spatial import cKDTree, distance_matrix, Delaunay
from math import cos
import pymap3d


def deg_to_decimal(degrees, minutes, seconds):
    return degrees + (minutes/60) + (seconds/3600)


def lat_to_km(lat):
    km = (40075.696/360) * cos(lat)
    return km * lat


def lon_to_km(lon):
    km = 40008.55/360
    return km * lon


'''
# Опорный пункт NSK1
B = deg_to_decimal(55, 60, 44.119)
L = deg_to_decimal(82, 59, 6.066)
H = 141.687
# 56.012255 82.985018 141.687
'''


def random_points(B, L, H, zone, amount):
    k = 1  # k - количество ближайших соседей для проверки расстояния
    start_point = {'Station': 'NSK1', 'B': B, 'L': L, 'H': H}  # исходный пункт

    points = [start_point]

    while len(points) < amount:
        Station = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 4))
        B = np.random.uniform(start_point['B'] - zone, start_point['B'] + zone)
        L = np.random.uniform(start_point['L'] - zone, start_point['L'] + zone)
        H = np.random.uniform(start_point['H'] - 25, start_point['H'] + 25)
        point = {'Station': Station, 'B': B, 'L': L, 'H': H}

        distances, indices = cKDTree(list([p['B'], p['L']] for p in points)).query([[B, L]], k=k)
        '''
        if len(points) < 6:
            if all(20000 < geodesic((p['B'], p['L']), (B, L)).meters < 30000 for p in points):
                print(f"Distances to nearest neighbors of {Station}: {distances}")
                start_point = point
                points.append(point)
        else:
            if sum(1 for p in points if 20000 < geodesic((p['B'], p['L']), (B, L)).meters < 30000) >= 6:
                print(f"Distances to nearest neighbors of {Station}: {distances}")
                start_point = point
                points.append(point)
                k += 1
        '''
        if all(20000 < geodesic((p['B'], p['L']), (B, L)).meters for p in points):
        #if any(20000 < geodesic((p['B'], p['L']), (B, L)).meters < 30000 for p in points)
        #if np.all(20 < d < 30 for d in distances):
            print(f"Distances to nearest neighbors of {Station}: {distances}")
            start_point = point
            points.append(point)
            k += 1

    return points


def triangulation(df):
    fig, ax = plt.subplots()

    # рисуем точки на графике
    ax.scatter(df['L'], df['B'], c='blue', zorder=3)
    for i, row in df.iterrows():
        ax.annotate(f"{row['H']:.0f} m\n{row['Station']}", (row['L'], row['B'] + 0.01), fontsize=10, color='blue')

    # строим геодезическую сеть триангуляции
    tri = Delaunay(df[['B', 'L']])
    edges = set()
    # получаем все ребра треугольников и добавляем их в множество edges
    for n in range(tri.nsimplex):
        simplex = tri.simplices[n]
        for i in range(3):
            for j in range(i, 3):
                if i != j:
                    edges.add((simplex[i], simplex[j]))

    # рисуем ребра треугольников на графике и добавляем текст с длиной линий
    edge_labels = set()
    for edge in edges:
        x1, y1 = df.iloc[edge[0]][['L', 'B']]
        x2, y2 = df.iloc[edge[1]][['L', 'B']]
        length = geodesic((y1, x1), (y2, x2)).meters
        if length <= 80000:  # проверяем длину линии
            label = f'{length:.0f} m'
            edge_labels.add((min(edge), max(edge), label))
            ax.plot([x1, x2], [y1, y2], c='black', linewidth=0.5, zorder=2)
        '''
        label = f'{geodesic((y1, x1), (y2, x2)).meters:.0f} m'
        edge_labels.add((min(edge), max(edge), label))
        ax.plot([x1, x2], [y1, y2], c='black', linewidth=0.5, zorder=2)
        '''
    # рисуем текст с информацией о длинах линий
    for label in edge_labels:
        edge = label[:2]
        label_text = label[2]
        x1, y1 = df.iloc[edge[0]][['L', 'B']]
        x2, y2 = df.iloc[edge[1]][['L', 'B']]
        label_x = (x1 + x2) / 2
        label_y = (y1 + y2) / 2
        rotation = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        if rotation > 90:
            rotation -= 180
        elif rotation < -90:
            rotation += 180
        bbox = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8, 'pad': 0.3}
        ax.annotate(label_text, (label_x, label_y), rotation=rotation, bbox=bbox, ha='center', va='center', fontsize=6,
                    color='red', zorder=4)

        '''
        dist = geodesic((y1, x1), (y2, x2)).meters
        ax.plot([x1, x2], [y1, y2], c='black', linewidth=0.5, zorder=2)
        ax.annotate(f"{dist:.0f} m", ((x1 + x2) / 2, (y1 + y2) / 2), fontsize=8, color='red')
        '''
    plt.show()


def my_geodetic2ecef(df):
    # создание нового DataFrame с обновленными координатами
    df_new = pd.DataFrame({'Station': df['Station'],
                           'X': 0,
                           'Y': 0,
                           'Z': 0})

    # перевод координат и обновление нового DataFrame
    for i, row in df.iterrows():
        X, Y, Z = pymap3d.geodetic2ecef(row['B'], row['L'], row['H'])
        df_new.at[i, 'X'] = X
        df_new.at[i, 'Y'] = Y
        df_new.at[i, 'Z'] = Z

    return df_new


random_network_data = pd.DataFrame(random_points(56.012255, 82.985018, 141.687, 0.5, 15))
#print(random_network_data)
#print(my_geodetic2ecef(random_network_data))
# df.to_csv('random_geodetic_points.csv', index=False)


# рисуем график
'''
plt.scatter(random_network_data['L'], random_network_data['B'], c='blue')

df_station = random_network_data.set_index('Station')
df_NSK1 = df_station.loc['NSK1']
df_NSK1 = df_NSK1.reset_index()
df_NSK1_L = df_NSK1['L']
df_NSK1_B = df_NSK1['B']
plt.scatter(df_NSK1_L, df_NSK1_B, c='red')
'''
triangulation(random_network_data)
plt.show()

'''
if system == 'ecef':
    ecef = pymap3d.geodetic2ecef(point['B'], point['L'], point['H'])
    point = {'Station': Station, 'X': ecef[0], 'Y': ecef[1], 'Z': ecef[2]}
elif system == 'blh':
    pass


x1, y1 = df.iloc[edge[0]][['L', 'B']]
        x2, y2 = df.iloc[edge[1]][['L', 'B']]
        ax.plot([x1, x2], [y1, y2], c='black', linewidth=0.5, zorder=2)
        
        # вычисляем центр линии
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        # вычисляем расстояние между точками
        dist = geodesic((y1, x1), (y2, x2)).meters

        # рассчитываем угол наклона текста в градусах
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # выводим текст вдоль линии
        ax.text(center_x, center_y, f'{dist:.0f} m', rotation=angle, ha='center', va='center')
       





'''