# 🏕️ 微缩造景贴纸屋 (DioramaSticker iOS App)

一款治愈系微缩手账「造景贴纸」原生 iOS App（SwiftUI · iOS 17+），附带零依赖浏览器端手机可试玩原型与自动化贴纸拆分管线。

## ✨ 已内置 4 大主题场景（共 89 张独立透明贴纸）

| 关卡 | 主题名称 | 贴纸数量 | 核心元素 |
| :---: | :--- | :---: | :--- |
| 01 | **🏕️ 森林篝火晚会** (`campfire_party`) | 30 张 | 11 个露营小人、6 只森林小动物、帐篷野餐垫、篝火炖锅、弯月星辰与萤火虫 |
| 02 | **🐱 猫咖甜品店** (`cat_cafe`) | 16 张 | 10 个店员与顾客、9 只软萌猫咪、豪华猫爬架、甜品柜蛋糕马卡龙、奶茶撸猫桌 |
| 03 | **🎆 海边夏日夜市** (`summer_market`) | 22 张 | 章鱼烧/刨冰/捞金鱼 3 大摊位、夏日萌娃、柴犬海鸥螃蟹海龟、遮阳伞与海边烟花 |
| 04 | **🎄 冬日圣诞小屋** (`winter_cabin`) | 21 张 | 温暖木屋壁炉、圣诞树与礼物堆、姜饼屋圆桌、堆雪人与滑雪橇、驯鹿企鹅哈士奇北极熊 |

## 🎮 核心功能

- **关卡图鉴大厅 (`LevelGalleryView`)**：2×2 卡片网格展示全部 4 大主题，独立记录每关进度与通关徽章。
- **闯关找位模式 (Classic Snap)**：
  - 白边模切贴纸卡槽 + 拇指防遮挡抬起拖拽偏移（`-46pt`）
  - 靠近目标位置触发磁吸脉冲光环、弹簧贴合、星光粒子绽放、iOS Haptic 震动反馈与治愈音效
  - 支持「轮廓提示」「按住看原图」「灯泡高亮定位」「双指缩放平移」
- **自由造景模式 (Free DIY)**：
  - 任意摆放贴纸（支持重复添加），点选后可自由**缩放、旋转、水平翻转、图层置顶/置底、复制、删除**
  - 一键载入官方布局二次创作，支持 `ImageRenderer` 导出 **896×1200 高清海报** 到系统相册

## 📂 仓库结构

```text
DioramaStickerApp/
├── iOS/DioramaSticker/               # 原生 SwiftUI iOS 17+ Xcode 工程（含 AppIcon 与全部 4 关素材）
│   ├── DioramaSticker.xcodeproj/     # 双击直接用 Xcode 15+ 打开编译
│   ├── project.yml                   # XcodeGen 配置
│   └── DioramaSticker/               # Models / Services / ViewModels / Views / Resources
├── web_prototype/
│   └── index.html                    # 单文件 4 关可试玩原型（打开浏览器即可在手机/电脑试玩）
└── assets_pipeline/
    ├── extract_stickers.py           # 森林篝火晚会三层无遮挡拆图脚本
    └── extract_new_levels.py         # 猫咖 / 夏日夜市 / 圣诞小屋自动化拆图脚本
```

## 🚀 在 Mac / iPhone 上运行

1. 克隆仓库到本地：
   ```bash
   git clone <your-repo-url>
   cd DioramaStickerApp/iOS/DioramaSticker
   open DioramaSticker.xcodeproj
   ```
2. 在 Xcode 中选择 iPhone 15/16 模拟器或连接真机（在 `Signing & Capabilities` 选择个人 Apple ID Team）。
3. 按 `Cmd + R` 运行，即可体验完整的 4 大造景关卡！
4. 如需立即在浏览器试玩，直接打开 `web_prototype/index.html`。
