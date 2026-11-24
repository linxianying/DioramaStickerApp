import UIKit

final class LevelLoader {
    static let shared = LevelLoader()
    private var imageCache: [String: UIImage] = [:]

    private init() {}

    func loadManifest() -> [LevelManifestItem] {
        let subdirs = ["Levels", "Resources/Levels", nil]
        for subdir in subdirs {
            if let url = Bundle.main.url(forResource: "levels_manifest", withExtension: "json", subdirectory: subdir),
               let data = try? Data(contentsOf: url),
               let manifest = try? JSONDecoder().decode(LevelManifest.self, from: data) {
                return manifest.levels
            }
        }
        // Fallback built-in manifest
        return [
            LevelManifestItem(id: "campfire_party", folder: "campfire_party", jsonFile: "level_campfire", emoji: "🏕️", titleZh: "森林篝火晚会", subtitleZh: "星空萤火与露营好友", accentHex: "#FFD86B"),
            LevelManifestItem(id: "cat_cafe", folder: "cat_cafe", jsonFile: "level_cat_cafe", emoji: "🐱", titleZh: "猫咖甜品店", subtitleZh: "草莓蛋糕与软萌猫咪", accentHex: "#FF9EB5"),
            LevelManifestItem(id: "summer_market", folder: "summer_market", jsonFile: "level_summer_market", emoji: "🎆", titleZh: "海边夏日夜市", subtitleZh: "烟花海风与美食摊位", accentHex: "#6CE0FF"),
            LevelManifestItem(id: "winter_cabin", folder: "winter_cabin", jsonFile: "level_winter_cabin", emoji: "🎄", titleZh: "冬日圣诞小屋", subtitleZh: "壁炉暖光与雪地派对", accentHex: "#FF8080"),
            LevelManifestItem(id: "bagel_shop", folder: "bagel_shop", jsonFile: "level_bagel_shop", emoji: "🥯", titleZh: "贝果铺货小店", subtitleZh: "新鲜贝果补货与奶酪香", accentHex: "#FFB86C"),
            LevelManifestItem(id: "bakery_shop", folder: "bakery_shop", jsonFile: "level_bakery_shop", emoji: "🥐", titleZh: "法式面包烘焙店", subtitleZh: "窑炉欧包与酥脆可颂", accentHex: "#F5B971")
        ]
    }

    func loadLevel(folderName: String, jsonName: String) -> LevelDefinition? {
        let subdirectories = [
            "Levels/\(folderName)",
            "Resources/Levels/\(folderName)",
            folderName,
            nil
        ]

        for subdir in subdirectories {
            if let url = Bundle.main.url(forResource: jsonName, withExtension: "json", subdirectory: subdir),
               let data = try? Data(contentsOf: url),
               let level = try? JSONDecoder().decode(LevelDefinition.self, from: data) {
                return level
            }
        }
        return nil
    }

    func loadImage(named filename: String, folderName: String) -> UIImage? {
        let cacheKey = "\(folderName)/\(filename)"
        if let cached = imageCache[cacheKey] {
            return cached
        }

        let name = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension.isEmpty ? "png" : (filename as NSString).pathExtension

        let subdirectories = [
            "Levels/\(folderName)",
            "Resources/Levels/\(folderName)",
            folderName,
            nil
        ]

        for subdir in subdirectories {
            if let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: subdir),
               let image = UIImage(contentsOfFile: url.path) {
                imageCache[cacheKey] = image
                return image
            }
        }

        return nil
    }
}
