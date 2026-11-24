# 森林篝火晚会 · 造景贴纸 iOS App (SwiftUI)

原生 iOS 17+ SwiftUI 造景贴纸应用，包含完整「**闯关找位**」与「**自由造景 DIY**」双模式，以及「森林篝火晚会」全套 30 张独立透明 PNG 贴纸与归一化坐标配置。

## 目录结构

```text
DioramaSticker/
├── DioramaSticker.xcodeproj/         # 可直接用 Xcode 15+ 打开运行
├── project.yml                       # XcodeGen 配置文件（可选）
└── DioramaSticker/
    ├── DioramaStickerApp.swift       # App 入口
    ├── Info.plist                    # 竖屏配置 + 相册保存权限声明
    ├── Assets.xcassets/              # AccentColor & AppIcon
    ├── Models/
    │   └── DioramaModels.swift       # 关卡、贴纸、自由造景实例与分类模型
    ├── Services/
    │   ├── LevelLoader.swift         # JSON 关卡与 PNG 贴纸内存缓存加载器
    │   ├── HapticManager.swift       # iOS 震动反馈（揭起/磁吸/落位/通关）
    │   └── SoundManager.swift        # AVAudioEngine 零依赖程序合成治愈音效
    ├── ViewModels/
    │   └── DioramaGameViewModel.swift# 双模式状态机、磁吸判定、缩放坐标换算与存档
    ├── Views/
    │   ├── MainGameView.swift        # 顶栏 HUD、浮动拖拽贴纸、通关弹窗、海报导出
    │   ├── DioramaCanvasView.swift   # 支持双指缩放/拖拽平移、轮廓提示、偷看原图、星光粒子
    │   ├── StickerTrayView.swift     # 分类筛选胶囊 + 白边立体贴纸卡槽
    │   └── FreeModeToolbarView.swift # 自由模式大小/旋转/翻转/置顶置底/复制/删除工具条
    └── Resources/
        └── Levels/campfire_party/
            ├── level_campfire.json   # 30 张贴纸精确归一化坐标与图层 zIndex
            ├── background.png        # 空白森林草地底图 (896×1200)
            ├── reference_complete.png# 官方合成参考原图
            └── *.png                 # 30 张透明底 PNG 贴纸
```

## 快速在 Xcode 中运行

1. 在 macOS 上安装 **Xcode 15+**。
2. 双击打开 `DioramaSticker.xcodeproj`（或运行 `xcodegen generate` 重新生成工程）。
3. 选择任意 iPhone 模拟器（推荐 iPhone 15 / 16 Pro）或连接真机（在 Signing & Capabilities 中选择自己的 Apple ID 个人团队即可真机调试）。
4. 按 `Cmd + R` 运行。

## 后续扩展更多关卡

只需在 `Resources/Levels/<new_level_id>/` 放入：
- `background.png`（无贴纸的纯场景底图）
- 各元素透明底 `.png` 贴纸
- `level_<id>.json`（遵循 `LevelDefinition` 结构，使用 `0...1` 归一化坐标，自动适配全部 iPhone/iPad 屏幕尺寸）

无需修改任何核心交互代码，即可直接新增「冬日小屋」「夏日海滩」「猫咖甜品店」等新关卡！
