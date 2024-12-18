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
    36.625, 69.556, 99.844, 128.27, 154.04, 174.57, 195.29, 212.25, 228.06, 242.74,
    257.45, 266.87, 275.82, 286.45, 290.25, 295.94, 299.42, 302.62, 307.57, 307.32
]
times_v = [
    65.945, 71.406, 84.486, 91.818, 109.06, 108.97, 109.58, 108.68, 116.75, 116.59,
    116.71, 116.11, 116.13, 116.44, 115.62, 142.70, 142.32, 142.96, 142.67, 143.34
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
