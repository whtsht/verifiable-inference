import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = './NotoSansJP-Regular.ttf'
font_manager.fontManager.addfont(font_path)
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.size'] = 13

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 2
plt.rcParams['lines.markeredgewidth'] = 2

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

figsize = (4,4)

# 実験1

image_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
times_n = [1.9237, 9.9936, 23.606, 43.145, 68.936, 99.678, 137.16, 180.57, 229.48, 285.68]

times_v = [20.857, 39.492, 50.127, 64.723, 70.843, 85.474, 95.083, 114.93, 125.65, 154.90]

plt.figure(figsize=figsize)
plt.plot(image_sizes, times_n, marker='o', color='blue', markersize=6)
plt.xlabel("画像サイズ [px]", weight='bold')
plt.ylabel("実行時間 [µs]")
plt.tight_layout()
plt.savefig('n__image_size.png', dpi=200, format='png')
plt.close()

plt.figure(figsize=figsize)
plt.plot(image_sizes, times_v, marker='s', color='red', markersize=6)
plt.xlabel("画像サイズ [px]")
plt.ylabel("実行時間 [ms]")
plt.tight_layout()
plt.savefig('v__image_size.png', dpi=200, format='png')
plt.close()


# 実験2

layers = list(range(1, 21))
times_n = [
  0.80273, 1.4881, 2.1202, 2.8509, 3.5942, 4.3412, 5.2266, 6.1322, 6.7869, 6.6215, 7.3355, 9.2459, 10.005, 8.7236, 9.2522, 9.7964, 10.433, 11.014, 11.632, 12.267,
]
times_v = [
  12.947, 15.416, 17.034, 18.463, 20.894, 20.766, 22.603, 22.595, 22.526, 25.589, 25.524, 25.355, 25.473, 27.650, 27.555, 27.727, 27.571, 27.763, 27.624, 27.601,
]

plt.figure(figsize=figsize)
plt.plot(layers, times_n, marker='o', color='blue', markersize=6)
plt.xlabel("畳み込み層の数")
plt.ylabel("実行時間 [µs]")
plt.tight_layout()
plt.savefig('n__layer_size.png', dpi=200, format='png')
plt.close()

plt.figure(figsize=figsize)
plt.plot(layers, times_v, marker='s', color='red', markersize=6)
plt.xlabel("畳み込み層の数")
plt.ylabel("実行時間 [ms]")
plt.tight_layout()
plt.savefig('v__layer_size.png', dpi=200, format='png')
plt.close()
