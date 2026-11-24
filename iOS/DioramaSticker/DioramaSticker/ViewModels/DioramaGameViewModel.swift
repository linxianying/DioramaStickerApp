import SwiftUI
import Combine

@MainActor
final class DioramaGameViewModel: ObservableObject {
    // MARK: - Multi-Level Catalog
    @Published private(set) var manifest: [LevelManifestItem] = []
    @Published private(set) var activeManifestItem: LevelManifestItem?
    @Published private(set) var level: LevelDefinition?
    @Published var gameMode: GameMode = .classic
    @Published var selectedCategory: StickerCategory = .all

    // MARK: - Classic Puzzle State
    @Published var placedStickerIDs: Set<String> = []
    @Published var showOutlineHints: Bool = true
    @Published var isPeekingReference: Bool = false
    @Published var spotlightStickerID: String? = nil
    @Published var hasUnlockedFreeMode: Bool = false
    @Published var showCompletionBanner: Bool = false

    // MARK: - Free DIY Mode State
    @Published var freePlacedStickers: [FreePlacedSticker] = []
    @Published var selectedFreeInstanceID: UUID? = nil

    // MARK: - Shared Interaction & Camera State
    @Published var activeDrag: ActiveDragState? = nil
    @Published var canvasGlobalFrame: CGRect = .zero
    @Published var zoomScale: CGFloat = 1.0
    @Published var panOffset: CGSize = .zero
    @Published var sparkles: [SparkleBurst] = []
    @Published var toastMessage: String? = nil
    @Published var isMuted: Bool = false {
        didSet { SoundManager.shared.isMuted = isMuted }
    }

    let snapThreshold: CGFloat = 0.082
    let dragFingerOffsetY: CGFloat = -46.0

    var currentFolderName: String {
        activeManifestItem?.folder ?? "campfire_party"
    }

    init() {
        manifest = LevelLoader.shared.loadManifest()
        if let first = manifest.first {
            selectLevel(first)
        }
    }

    func selectLevel(_ item: LevelManifestItem) {
        activeManifestItem = item
        level = LevelLoader.shared.loadLevel(folderName: item.folder, jsonName: item.jsonFile)
        selectedCategory = .all
        spotlightStickerID = nil
        showCompletionBanner = false
        selectedFreeInstanceID = nil
        zoomScale = 1.0
        panOffset = .zero
        restoreProgress(for: item.id)
    }

    // MARK: - Per-Level Progress Summary for Gallery
    func savedPlacedCount(for levelId: String) -> Int {
        if activeManifestItem?.id == levelId {
            return placedStickerIDs.count
        }
        return UserDefaults.standard.stringArray(forKey: classicKey(levelId))?.count ?? 0
    }

    func totalStickerCount(for item: LevelManifestItem) -> Int {
        if activeManifestItem?.id == item.id, let lvl = level {
            return lvl.stickers.count
        }
        return LevelLoader.shared.loadLevel(folderName: item.folder, jsonName: item.jsonFile)?.stickers.count ?? 0
    }

    // MARK: - Derived Lists
    var totalStickerCount: Int {
        level?.stickers.count ?? 0
    }

    var placedCount: Int {
        placedStickerIDs.count
    }

    var progressFraction: Double {
        guard totalStickerCount > 0 else { return 0 }
        return Double(placedCount) / Double(totalStickerCount)
    }

    var isClassicComplete: Bool {
        totalStickerCount > 0 && placedCount == totalStickerCount
    }

    /// Only display categories that actually have stickers in the current level
    var availableCategories: [StickerCategory] {
        guard let all = level?.stickers else { return [.all] }
        let present = Set(all.map(\.category))
        return StickerCategory.allCases.filter { $0 == .all || present.contains($0.rawValue) }
    }

    var trayStickers: [StickerDefinition] {
        guard let all = level?.stickers else { return [] }
        let base = (gameMode == .classic) ? all.filter { !placedStickerIDs.contains($0.id) } : all
        if selectedCategory == .all { return base }
        return base.filter { $0.category == selectedCategory.rawValue }
    }

    var placedClassicStickers: [StickerDefinition] {
        guard let all = level?.stickers else { return [] }
        return all.filter { placedStickerIDs.contains($0.id) }.sorted { $0.zIndex < $1.zIndex }
    }

    func stickerDefinition(for id: String) -> StickerDefinition? {
        level?.stickers.first { $0.id == id }
    }

    // MARK: - Coordinate Conversion
    func normalizedPoint(fromGlobal globalPoint: CGPoint) -> CGPoint? {
        guard canvasGlobalFrame.width > 1, canvasGlobalFrame.height > 1 else { return nil }
        let offsetY = (activeDrag?.freeInstanceId == nil) ? dragFingerOffsetY : 0
        let adjusted = CGPoint(x: globalPoint.x, y: globalPoint.y + offsetY)
        let centerX = canvasGlobalFrame.midX + panOffset.width
        let centerY = canvasGlobalFrame.midY + panOffset.height

        let localFromCenterX = (adjusted.x - centerX) / zoomScale
        let localFromCenterY = (adjusted.y - centerY) / zoomScale

        return CGPoint(
            x: (localFromCenterX / canvasGlobalFrame.width) + 0.5,
            y: (localFromCenterY / canvasGlobalFrame.height) + 0.5
        )
    }

    // MARK: - Drag Lifecycle
    func beginDrag(sticker: StickerDefinition, at globalPoint: CGPoint, freeInstanceId: UUID? = nil) {
        if activeDrag == nil {
            HapticManager.shared.stickerPickup()
            SoundManager.shared.playPickup()
        }
        var state = ActiveDragState(sticker: sticker, globalLocation: globalPoint, isNearTarget: false, freeInstanceId: freeInstanceId)
        activeDrag = state
        if gameMode == .classic, let norm = normalizedPoint(fromGlobal: globalPoint) {
            state.isNearTarget = isWithinSnapZone(sticker: sticker, normPoint: norm)
            activeDrag = state
        }
    }

    func updateDrag(to globalPoint: CGPoint) {
        guard var current = activeDrag else { return }
        current.globalLocation = globalPoint

        if gameMode == .classic, let norm = normalizedPoint(fromGlobal: globalPoint) {
            let near = isWithinSnapZone(sticker: current.sticker, normPoint: norm)
            if near && !current.isNearTarget {
                HapticManager.shared.magneticEnter()
            }
            current.isNearTarget = near
        }
        activeDrag = current
    }

    func endDrag(at globalPoint: CGPoint) {
        guard let current = activeDrag else { return }
        let norm = normalizedPoint(fromGlobal: globalPoint)
        activeDrag = nil

        guard let norm else { return }

        if gameMode == .classic {
            if isWithinSnapZone(sticker: current.sticker, normPoint: norm) {
                snapClassicSticker(current.sticker)
            } else {
                HapticManager.shared.stickerReturn()
                SoundManager.shared.playReturn()
            }
        } else {
            guard norm.x >= 0.02, norm.x <= 0.98, norm.y >= 0.02, norm.y <= 0.98 else {
                HapticManager.shared.stickerReturn()
                return
            }

            if let existingID = current.freeInstanceId,
               let idx = freePlacedStickers.firstIndex(where: { $0.instanceId == existingID }) {
                withAnimation(.spring(response: 0.28, dampingFraction: 0.72)) {
                    freePlacedStickers[idx].normX = norm.x
                    freePlacedStickers[idx].normY = norm.y
                    selectedFreeInstanceID = existingID
                }
            } else {
                let nextZ = (freePlacedStickers.map(\.zIndex).max() ?? 10) + 1
                let instance = FreePlacedSticker(
                    instanceId: UUID(),
                    stickerId: current.sticker.id,
                    normX: norm.x,
                    normY: norm.y,
                    scale: 1.0,
                    rotationDegrees: 0,
                    isFlippedHorizontally: false,
                    zIndex: nextZ
                )
                withAnimation(.spring(response: 0.32, dampingFraction: 0.68)) {
                    freePlacedStickers.append(instance)
                    selectedFreeInstanceID = instance.instanceId
                }
            }
            HapticManager.shared.stickerSnap()
            SoundManager.shared.playSnap()
            triggerSparkle(atX: norm.x, y: norm.y)
            persistProgress()
        }
    }

    private func isWithinSnapZone(sticker: StickerDefinition, normPoint: CGPoint) -> Bool {
        let bonus = max(sticker.normW, sticker.normH) > 0.42 ? 0.14 : 0.0
        return hypot(normPoint.x - sticker.normX, normPoint.y - sticker.normY) <= (snapThreshold + bonus)
    }

    private func snapClassicSticker(_ sticker: StickerDefinition) {
        withAnimation(.spring(response: 0.32, dampingFraction: 0.62)) {
            placedStickerIDs.insert(sticker.id)
            if spotlightStickerID == sticker.id { spotlightStickerID = nil }
        }
        HapticManager.shared.stickerSnap()
        SoundManager.shared.playSnap()
        triggerSparkle(atX: sticker.normX, y: sticker.normY)

        if isClassicComplete {
            hasUnlockedFreeMode = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                HapticManager.shared.levelComplete()
                SoundManager.shared.playCompleteFanfare()
                withAnimation(.spring(response: 0.45, dampingFraction: 0.72)) {
                    self.showCompletionBanner = true
                }
            }
        }
        persistProgress()
    }

    // MARK: - Hints & Controls
    func triggerSpotlightHint() {
        guard let next = trayStickers.first ?? level?.stickers.first(where: { !placedStickerIDs.contains($0.id) }) else {
            return
        }
        HapticManager.shared.magneticEnter()
        withAnimation(.easeInOut(duration: 0.25)) {
            spotlightStickerID = next.id
        }
        showToast("已高亮「\(next.nameZh)」的目标位置 ✨")

        DispatchQueue.main.asyncAfter(deadline: .now() + 3.8) { [weak self] in
            if self?.spotlightStickerID == next.id {
                withAnimation(.easeOut(duration: 0.3)) {
                    self?.spotlightStickerID = nil
                }
            }
        }
    }

    func resetCamera() {
        withAnimation(.spring(response: 0.35, dampingFraction: 0.78)) {
            zoomScale = 1.0
            panOffset = .zero
        }
    }

    func resetCurrentMode() {
        HapticManager.shared.stickerReturn()
        if gameMode == .classic {
            withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
                placedStickerIDs.removeAll()
                spotlightStickerID = nil
                showCompletionBanner = false
            }
            showToast("已重置本关闯关进度")
        } else {
            withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
                freePlacedStickers.removeAll()
                selectedFreeInstanceID = nil
            }
            showToast("已清空自由造景画布")
        }
        persistProgress()
    }

    func loadOfficialLayoutIntoFreeMode() {
        guard let stickers = level?.stickers else { return }
        withAnimation(.spring(response: 0.4, dampingFraction: 0.75)) {
            freePlacedStickers = stickers.map {
                FreePlacedSticker(
                    instanceId: UUID(),
                    stickerId: $0.id,
                    normX: $0.normX,
                    normY: $0.normY,
                    scale: 1.0,
                    rotationDegrees: 0,
                    isFlippedHorizontally: false,
                    zIndex: $0.zIndex
                )
            }
            selectedFreeInstanceID = nil
        }
        HapticManager.shared.stickerSnap()
        showToast("已载入官方布局，可任意二次创作！")
        persistProgress()
    }

    // MARK: - Free Mode Manipulation
    var selectedFreeSticker: FreePlacedSticker? {
        guard let id = selectedFreeInstanceID else { return nil }
        return freePlacedStickers.first { $0.instanceId == id }
    }

    func updateSelectedFreeSticker(_ mutate: (inout FreePlacedSticker) -> Void) {
        guard let id = selectedFreeInstanceID,
              let idx = freePlacedStickers.firstIndex(where: { $0.instanceId == id }) else { return }
        mutate(&freePlacedStickers[idx])
        persistProgress()
    }

    func bringSelectedForward() {
        let maxZ = freePlacedStickers.map(\.zIndex).max() ?? 10
        updateSelectedFreeSticker { $0.zIndex = maxZ + 1 }
        HapticManager.shared.magneticEnter()
    }

    func sendSelectedBackward() {
        let minZ = freePlacedStickers.map(\.zIndex).min() ?? 10
        updateSelectedFreeSticker { $0.zIndex = minZ - 1 }
        HapticManager.shared.magneticEnter()
    }

    func duplicateSelectedFreeSticker() {
        guard let current = selectedFreeSticker else { return }
        let nextZ = (freePlacedStickers.map(\.zIndex).max() ?? 10) + 1
        let copy = FreePlacedSticker(
            instanceId: UUID(),
            stickerId: current.stickerId,
            normX: min(0.92, current.normX + 0.05),
            normY: min(0.92, current.normY + 0.04),
            scale: current.scale,
            rotationDegrees: current.rotationDegrees,
            isFlippedHorizontally: current.isFlippedHorizontally,
            zIndex: nextZ
        )
        withAnimation(.spring(response: 0.28, dampingFraction: 0.7)) {
            freePlacedStickers.append(copy)
            selectedFreeInstanceID = copy.instanceId
        }
        HapticManager.shared.stickerPickup()
        persistProgress()
    }

    func deleteSelectedFreeSticker() {
        guard let id = selectedFreeInstanceID else { return }
        withAnimation(.spring(response: 0.25, dampingFraction: 0.75)) {
            freePlacedStickers.removeAll { $0.instanceId == id }
            selectedFreeInstanceID = nil
        }
        HapticManager.shared.stickerReturn()
        persistProgress()
    }

    // MARK: - Sparkles & Toast
    private func triggerSparkle(atX x: CGFloat, y: CGFloat) {
        let burst = SparkleBurst(normX: x, normY: y)
        sparkles.append(burst)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.9) { [weak self] in
            self?.sparkles.removeAll { $0.id == burst.id }
        }
    }

    func showToast(_ message: String) {
        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
            toastMessage = message
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.2) { [weak self] in
            if self?.toastMessage == message {
                withAnimation(.easeOut(duration: 0.25)) {
                    self?.toastMessage = nil
                }
            }
        }
    }

    // MARK: - Per-Level Persistence
    private func classicKey(_ levelId: String) -> String { "diorama_classic_\(levelId)_v1" }
    private func unlockedKey(_ levelId: String) -> String { "diorama_unlocked_\(levelId)_v1" }
    private func freeKey(_ levelId: String) -> String { "diorama_free_\(levelId)_v1" }

    private func persistProgress() {
        guard let lid = activeManifestItem?.id else { return }
        UserDefaults.standard.set(Array(placedStickerIDs), forKey: classicKey(lid))
        UserDefaults.standard.set(hasUnlockedFreeMode, forKey: unlockedKey(lid))
        if let data = try? JSONEncoder().encode(freePlacedStickers) {
            UserDefaults.standard.set(data, forKey: freeKey(lid))
        }
    }

    private func restoreProgress(for levelId: String) {
        if let saved = UserDefaults.standard.stringArray(forKey: classicKey(levelId)) {
            placedStickerIDs = Set(saved)
        } else {
            placedStickerIDs = []
        }
        hasUnlockedFreeMode = UserDefaults.standard.bool(forKey: unlockedKey(levelId)) || isClassicComplete
        if let data = UserDefaults.standard.data(forKey: freeKey(levelId)),
           let decoded = try? JSONDecoder().decode([FreePlacedSticker].self, from: data) {
            freePlacedStickers = decoded
        } else {
            freePlacedStickers = []
        }
    }
}
