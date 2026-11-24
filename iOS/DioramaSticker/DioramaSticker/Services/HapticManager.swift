import UIKit

final class HapticManager {
    static let shared = HapticManager()

    var isEnabled: Bool = true

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

    func stickerPickup() {
        guard isEnabled else { return }
        lightImpact.impactOccurred(intensity: 0.7)
    }

    func magneticEnter() {
        guard isEnabled else { return }
        selection.selectionChanged()
    }

    func stickerSnap() {
        guard isEnabled else { return }
        mediumImpact.impactOccurred(intensity: 0.9)
    }

    func stickerReturn() {
        guard isEnabled else { return }
        softImpact.impactOccurred(intensity: 0.5)
    }

    func levelComplete() {
        guard isEnabled else { return }
        notification.notificationOccurred(.success)
    }
}
