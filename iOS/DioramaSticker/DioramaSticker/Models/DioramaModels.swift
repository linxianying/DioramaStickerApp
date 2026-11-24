import Foundation
import CoreGraphics

struct LevelManifest: Codable {
    let levels: [LevelManifestItem]
}

struct LevelManifestItem: Codable, Identifiable, Hashable {
    let id: String
    let folder: String
    let jsonFile: String
    let emoji: String
    let titleZh: String
    let subtitleZh: String
    let accentHex: String

    enum CodingKeys: String, CodingKey {
        case id, folder, jsonFile, emoji, accentHex
        case titleZh = "title_zh"
        case subtitleZh = "subtitle_zh"
    }
}

enum GameMode: String, CaseIterable, Identifiable {
    case classic = "classic"
    case freeDIY = "freeDIY"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .classic: return "闯关找位"
        case .freeDIY: return "自由造景"
        }
    }

    var systemImage: String {
        switch self {
        case .classic: return "sparkles.rectangle.stack"
        case .freeDIY: return "wand.and.stars"
        }
    }
}

enum StickerCategory: String, CaseIterable, Identifiable {
    case all = "全部"
    case camp = "营地设施"
    case props = "露营道具"
    case characters = "卡通人物"
    case animals = "可爱动物"
    case sky = "星空光效"

    var id: String { rawValue }

    var displayLabel: String {
        switch self {
        case .all: return "全部"
        case .camp: return "场景家具"
        case .props: return "可爱道具"
        case .characters: return "卡通人物"
        case .animals: return "萌宠动物"
        case .sky: return "星空光效"
        }
    }

    var icon: String {
        switch self {
        case .all: return "✨"
        case .camp: return "⛺"
        case .props: return "🎀"
        case .characters: return "🧒"
        case .animals: return "🐾"
        case .sky: return "🌙"
        }
    }
}

struct LevelDefinition: Codable, Identifiable {
    let levelId: String
    let titleZh: String
    let titleEn: String
    let subtitleZh: String?
    let canvasWidth: CGFloat
    let canvasHeight: CGFloat
    let backgroundFile: String
    let stickers: [StickerDefinition]

    var id: String { levelId }

    var aspectRatio: CGFloat {
        canvasWidth / canvasHeight
    }

    enum CodingKeys: String, CodingKey {
        case levelId
        case titleZh = "title_zh"
        case titleEn = "title_en"
        case subtitleZh = "subtitle_zh"
        case canvasWidth
        case canvasHeight
        case backgroundFile
        case stickers
    }
}

struct StickerDefinition: Codable, Identifiable, Hashable {
    let id: String
    let file: String
    let nameZh: String
    let nameEn: String
    let category: String
    let zIndex: Int
    let x: CGFloat
    let y: CGFloat
    let width: CGFloat
    let height: CGFloat
    let normX: CGFloat
    let normY: CGFloat
    let normW: CGFloat
    let normH: CGFloat

    enum CodingKeys: String, CodingKey {
        case id, file, category, zIndex, x, y, width, height, normX, normY, normW, normH
        case nameZh = "name_zh"
        case nameEn = "name_en"
    }
}

struct FreePlacedSticker: Identifiable, Codable, Equatable {
    let instanceId: UUID
    let stickerId: String
    var normX: CGFloat
    var normY: CGFloat
    var scale: CGFloat
    var rotationDegrees: Double
    var isFlippedHorizontally: Bool
    var zIndex: Int

    var id: UUID { instanceId }
}

struct ActiveDragState: Equatable {
    let sticker: StickerDefinition
    var globalLocation: CGPoint
    var isNearTarget: Bool = false
    var freeInstanceId: UUID? = nil
}

struct SparkleBurst: Identifiable {
    let id = UUID()
    let normX: CGFloat
    let normY: CGFloat
    let createdAt = Date()
}
