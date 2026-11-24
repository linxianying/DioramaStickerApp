import UIKit

final class HapticManager {
    static let shared = HapticManager()

    private let lightImpact = UIImpactFeedbackGenerator(style: .light)
    private let mediumImpact = UIImpactFeedbackGenerator(style: .medium)
    private let softImpact = UIImpactFeedbackGenerator(style: .soft)
    private let notification = UINotificationFeedbackGenerator()
    private let selection = UISelectionFeedbackGenerator()

    private init() {
        prepare()
    }

    func prepare() {
        lightImpact.prepare()
        mediumImpact.prepare()
        softImpact.prepare()
        notification.prepare()
        selection.prepare()
    }

    /// Subtle paper peel feel when picking up a sticker from the tray
    func stickerPickup() {
        lightImpact.impactOccurred(intensity: 0.7)
    }

    /// Gentle tick when dragging into the magnetic snap zone
    func magneticEnter() {
        selection.selectionChanged()
    }

    /// Juicy tactile snap when sticker locks into place
    func stickerSnap() {
        mediumImpact.impactOccurred(intensity: 0.9)
    }

    /// Soft bounce when sticker returns to tray (miss)
    func stickerReturn() {
        softImpact.impactOccurred(intensity: 0.5)
    }

    /// Celebration haptic on level completion
    func levelComplete() {
        notification.notificationOccurred(.success)
    }
}
