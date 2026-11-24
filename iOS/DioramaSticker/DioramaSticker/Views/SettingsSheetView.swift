import SwiftUI

struct SettingsSheetView: View {
    @ObservedObject var viewModel: DioramaGameViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ZStack {
                Color(red: 0.10, green: 0.12, blue: 0.22).ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 16) {
                        // Section 1: Pickup Tool Mode (Tweezers vs Finger)
                        settingsCard(title: "贴纸拿取方式", icon: "hand.draw") {
                            VStack(spacing: 10) {
                                ForEach(PickupToolStyle.allCases) { style in
                                    let active = viewModel.pickupToolStyle == style
                                    Button {
                                        withAnimation(.spring(response: 0.25, dampingFraction: 0.78)) {
                                            viewModel.pickupToolStyle = style
                                        }
                                        HapticManager.shared.magneticEnter()
                                    } label: {
                                        HStack(spacing: 12) {
                                            Text(style == .tweezers ? "✂️" : "🖐️")
                                                .font(.system(size: 22))
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text(style.title)
                                                    .font(.system(size: 14.5, weight: .bold))
                                                    .foregroundStyle(.white)
                                                Text(style.subtitle)
                                                    .font(.system(size: 11.5))
                                                    .foregroundStyle(.white.opacity(0.68))
                                            }
                                            Spacer()
                                            Image(systemName: active ? "checkmark.circle.fill" : "circle")
                                                .font(.system(size: 20))
                                                .foregroundStyle(active ? Color(red: 1.0, green: 0.84, blue: 0.42) : .white.opacity(0.3))
                                        }
                                        .padding(12)
                                        .background(
                                            RoundedRectangle(cornerRadius: 16)
                                                .fill(active ? Color(red: 1.0, green: 0.84, blue: 0.42).opacity(0.16) : Color.white.opacity(0.06))
                                                .overlay(
                                                    RoundedRectangle(cornerRadius: 16)
                                                        .strokeBorder(active ? Color(red: 1.0, green: 0.84, blue: 0.42).opacity(0.6) : .clear, lineWidth: 1.2)
                                                )
                                        )
                                    }
                                    .buttonStyle(.plain)
                                }

                                if viewModel.pickupToolStyle == .tweezers {
                                    HStack(spacing: 10) {
                                        Text("镊子配色")
                                            .font(.system(size: 12.5, weight: .semibold))
                                            .foregroundStyle(.white.opacity(0.75))
                                        Spacer()
                                        ForEach(TweezerTheme.allCases) { theme in
                                            let selected = viewModel.tweezerTheme == theme
                                            Button {
                                                viewModel.tweezerTheme = theme
                                                HapticManager.shared.magneticEnter()
                                            } label: {
                                                HStack(spacing: 5) {
                                                    Circle()
                                                        .fill(theme.primaryColor)
                                                        .frame(width: 12, height: 12)
                                                    Text(theme.title)
                                                        .font(.system(size: 11.5, weight: .semibold))
                                                }
                                                .foregroundStyle(selected ? Color(red: 0.14, green: 0.12, blue: 0.20) : .white.opacity(0.85))
                                                .padding(.horizontal, 10)
                                                .padding(.vertical, 6)
                                                .background(
                                                    Capsule().fill(selected ? Color(red: 1.0, green: 0.84, blue: 0.42) : Color.white.opacity(0.10))
                                                )
                                            }
                                            .buttonStyle(.plain)
                                        }
                                    }
                                    .padding(.top, 4)
                                }
                            }
                        }

                        // Section 2: Animations & Sensory Feedback
                        settingsCard(title: "动效与感官体验", icon: "sparkles") {
                            VStack(spacing: 14) {
                                Toggle(isOn: $viewModel.enablePeelAnimation) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("📄 贴纸撕纸剥离动画")
                                            .font(.system(size: 14, weight: .semibold))
                                            .foregroundStyle(.white)
                                        Text("从卡槽揭起贴纸时播放 3D 翻角卷起与撕纸沙沙声")
                                            .font(.system(size: 11.5))
                                            .foregroundStyle(.white.opacity(0.65))
                                    }
                                }
                                .tint(Color(red: 1.0, green: 0.84, blue: 0.42))

                                Divider().overlay(Color.white.opacity(0.1))

                                Toggle(isOn: Binding(
                                    get: { !viewModel.isMuted },
                                    set: { viewModel.isMuted = !$0 }
                                )) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("🔊 治愈交互音效")
                                            .font(.system(size: 14, weight: .semibold))
                                            .foregroundStyle(.white)
                                        Text("纸张剥离声、磁吸叮咚音与通关旋律")
                                            .font(.system(size: 11.5))
                                            .foregroundStyle(.white.opacity(0.65))
                                    }
                                }
                                .tint(Color(red: 1.0, green: 0.84, blue: 0.42))

                                Divider().overlay(Color.white.opacity(0.1))

                                Toggle(isOn: $viewModel.enableHaptics) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("📳 震动触感反馈 (Haptics)")
                                            .font(.system(size: 14, weight: .semibold))
                                            .foregroundStyle(.white)
                                        Text("揭起贴纸、进入磁吸区与落位时的细腻震动")
                                            .font(.system(size: 11.5))
                                            .foregroundStyle(.white.opacity(0.65))
                                    }
                                }
                                .tint(Color(red: 1.0, green: 0.84, blue: 0.42))
                            }
                        }
                    }
                    .padding(16)
                }
            }
            .navigationTitle("手感与动画设置")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                        .fontWeight(.bold)
                        .foregroundStyle(Color(red: 1.0, green: 0.84, blue: 0.42))
                }
            }
        }
    }

    private func settingsCard<Content: View>(title: String, icon: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: icon)
                .font(.system(size: 13.5, weight: .bold))
                .foregroundStyle(Color(red: 1.0, green: 0.86, blue: 0.45))

            content()
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(Color(red: 0.15, green: 0.18, blue: 0.30))
                .overlay(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.10), lineWidth: 1)
                )
        )
    }
}
