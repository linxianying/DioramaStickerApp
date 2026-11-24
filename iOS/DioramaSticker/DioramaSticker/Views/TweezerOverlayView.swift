import SwiftUI

enum PickupToolStyle: String, CaseIterable, Identifiable {
    case tweezers = "tweezers"
    case finger = "finger"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .tweezers: return "镊子夹取"
        case .finger: return "手指直贴"
        }
    }

    var subtitle: String {
        switch self {
        case .tweezers: return "模拟手账微缩镊子夹起贴纸，沉浸感拉满"
        case .finger: return "简洁直观，手指直接拖拽落位"
        }
    }
}

enum TweezerTheme: String, CaseIterable, Identifiable {
    case roseGold = "roseGold"
    case champagne = "champagne"
    case silver = "silver"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .roseGold: return "樱花粉"
        case .champagne: return "奶油金"
        case .silver: return "极简银"
        }
    }

    var primaryColor: Color {
        switch self {
        case .roseGold: return Color(red: 0.98, green: 0.68, blue: 0.76)
        case .champagne: return Color(red: 0.98, green: 0.85, blue: 0.56)
        case .silver: return Color(red: 0.84, green: 0.88, blue: 0.94)
        }
    }

    var highlightColor: Color {
        switch self {
        case .roseGold: return Color(red: 1.0, green: 0.88, blue: 0.92)
        case .champagne: return Color(red: 1.0, green: 0.95, blue: 0.82)
        case .silver: return Color.white
        }
    }

    var strokeColor: Color {
        switch self {
        case .roseGold: return Color(red: 0.55, green: 0.28, blue: 0.36)
        case .champagne: return Color(red: 0.55, green: 0.42, blue: 0.20)
        case .silver: return Color(red: 0.34, green: 0.38, blue: 0.46)
        }
    }
}

/// Cute angled precision diorama sticker tweezers that pinch the lifted sticker
struct TweezerOverlayView: View {
    let theme: TweezerTheme
    let isPinching: Bool
    let isNearTarget: Bool

    var body: some View {
        ZStack {
            // Back prong
            TweezerProngShape(spread: isPinching ? 0 : 7)
                .fill(
                    LinearGradient(
                        colors: [theme.primaryColor.opacity(0.85), theme.strokeColor.opacity(0.65)],
                        startPoint: .topTrailing,
                        endPoint: .bottomLeading
                    )
                )
                .overlay(
                    TweezerProngShape(spread: isPinching ? 0 : 7)
                        .stroke(theme.strokeColor, lineWidth: 1.5)
                )
                .offset(x: 4, y: 4)

            // Front prong with glossy highlight
            TweezerProngShape(spread: 0)
                .fill(
                    LinearGradient(
                        colors: [theme.highlightColor, theme.primaryColor],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    TweezerProngShape(spread: 0)
                        .stroke(theme.strokeColor, lineWidth: 1.6)
                )
        }
        .frame(width: 94, height: 94)
        .shadow(color: .black.opacity(0.38), radius: 7, x: 2, y: 6)
        .rotationEffect(.degrees(isNearTarget ? -6 : 0))
        .scaleEffect(isPinching ? 1.0 : 1.06)
        .animation(.spring(response: 0.22, dampingFraction: 0.68), value: isPinching)
        .animation(.easeOut(duration: 0.16), value: isNearTarget)
    }
}

private struct TweezerProngShape: Shape {
    var spread: CGFloat

    var animatableData: CGFloat {
        get { spread }
        set { spread = newValue }
    }

    func path(in rect: CGRect) -> Path {
        var p = Path()
        // Curved tip at bottom-left (0, rect.height) extending up-right to handle (rect.width, 0)
        p.move(to: CGPoint(x: 4, y: rect.height - 6 - spread))
        p.addQuadCurve(
            to: CGPoint(x: rect.width - 14, y: 6),
            control: CGPoint(x: rect.width * 0.44, y: rect.height * 0.42 - spread * 0.5)
        )
        p.addLine(to: CGPoint(x: rect.width - 4, y: 15))
        p.addQuadCurve(
            to: CGPoint(x: 8, y: rect.height - 2 - spread),
            control: CGPoint(x: rect.width * 0.52, y: rect.height * 0.52 - spread * 0.4)
        )
        p.closeSubpath()
        return p
    }
}

/// Animated 3D paper-peel curl & sheen modifier played when lifting a sticker from the tray
struct StickerPeelEffectModifier: ViewModifier {
    let enabled: Bool
    @State private var peelPhase: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .rotation3DEffect(
                .degrees(enabled ? Double((1 - peelPhase) * 28) : 0),
                axis: (x: 1, y: -0.6, z: 0),
                perspective: 0.55
            )
            .scaleEffect(enabled ? (0.86 + 0.14 * peelPhase) : 1.0)
            .overlay {
                if enabled && peelPhase < 0.99 {
                    LinearGradient(
                        colors: [.clear, .white.opacity(0.65 * (1 - peelPhase)), .clear],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                    .offset(x: (peelPhase - 0.5) * 90)
                    .mask(content)
                }
            }
            .onAppear {
                guard enabled else { peelPhase = 1; return }
                peelPhase = 0
                withAnimation(.spring(response: 0.30, dampingFraction: 0.72)) {
                    peelPhase = 1
                }
            }
    }
}
