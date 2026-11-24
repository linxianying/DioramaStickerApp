# 🏕️ 微缩造景贴纸屋 (DioramaSticker iOS App & Mobile Web PWA)

一款治愈系微缩手账「造景贴纸」原生 iOS App（SwiftUI · iOS 17+），附带 **GitHub Pages 手机扫码即玩 PWA 网页版** 与自动化贴纸拆分管线。

🌐 **手机 / 浏览器在线试玩**：https://linxianying.github.io/DioramaStickerApp/  
（iPhone Safari 打开后点击「分享 → 添加到主屏幕」即可全屏当原生 App 玩！）

## ✨ 已内置 6 大主题场景（共 121 张独立透明贴纸）

| 关卡 | 主题名称 | 贴纸数量 | 核心元素 |
| :---: | :--- | :---: | :--- |
| 01 | **🏕️ 森林篝火晚会** (`campfire_party`) | 30 张 | 11 个露营小人、6 只森林小动物、帐篷野餐垫、篝火炖锅、弯月星辰与萤火虫 |
| 02 | **🐱 猫咖甜品店** (`cat_cafe`) | 16 张 | 10 个店员与顾客、9 只软萌猫咪、豪华猫爬架、甜品柜蛋糕马卡龙、奶茶撸猫桌 |
| 03 | **🎆 海边夏日夜市** (`summer_market`) | 22 张 | 章鱼烧/刨冰/捞金鱼 3 大摊位、夏日萌娃、柴犬海鸥螃蟹海龟、遮阳伞与海边烟花 |
| 04 | **🎄 冬日圣诞小屋** (`winter_cabin`) | 21 张 | 温暖木屋壁炉、圣诞树与礼物堆、姜饼屋圆桌、堆雪人与滑雪橇、驯鹿企鹅哈士奇北极熊 |
| 05 | **🥯 贝果铺货小店** (`bagel_shop`) | 16 张 | 补货贝果师、抹奶酪/夹贝果店员、快递小哥、贝果货箱、小仓鼠/卷毛狗/小兔/小橘猫 |
| 06 | **🥐 法式面包烘焙店** (`bakery_shop`) | 16 张 | 砖窑老面包师、揉面小厨与迷你小老鼠、可颂推车、搬面粉学徒、奶奶与腊肠小狗/三花猫 |

## 🎮 核心功能与沉浸式手感设置

- **⚙️ 手感与动画设置面板（支持双端持久化保存）**：
  - **贴纸拿取方式**：可在 **✂️ 镊子夹取**（支持樱花粉 / 奶油金 / 极简银 3 款微缩弯曲镊子）与 **🖐️ 手指直贴** 之间自由切换
  - **📄 贴纸撕纸剥离动画**：可开关 3D 翻角卷起 + 对角高光扫过 + 程序合成 ASMR 撕纸沙沙声
  - **🔊 治愈音效** & **📳 触感震动反馈** 独立开关
- **闯关找位模式 (Classic Snap)**：轮廓提示、按住偷看原图、灯泡高亮定位、磁吸脉冲光环、星光粒子绽放、缩放平移。
- **自由造景模式 (Free DIY)**：任意重复摆放、大小缩放、旋转、水平翻转、图层置顶/置底、一键载入官方布局、导出 **896×1200 高清海报**。

## 📂 仓库结构

```text
DioramaStickerApp/
├── docs/                             # GitHub Pages 手机即玩 PWA 站点（按关懒加载 WebP）
├── iOS/DioramaSticker/               # 原生 SwiftUI iOS 17+ Xcode 工程（含 AppIcon 与 6 关素材）
├── web_prototype/                    # 离线单文件原型备份
└── assets_pipeline/                  # Python OpenCV 自动化拆图脚本
```

## 🚀 在 Mac / Xcode 运行原生 iOS 版

```bash
git clone https://github.com/linxianying/DioramaStickerApp.git
cd DioramaStickerApp/iOS/DioramaSticker
open DioramaSticker.xcodeproj
```
在 Xcode 中选择 iPhone 模拟器或真机按 `Cmd + R` 即可运行。
