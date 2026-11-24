import SwiftUI

struct FreeModeToolbarView: View {
    @ObservedObject var viewModel: DioramaGameViewModel
    let onExportPhoto: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            if let selected = viewModel.selectedFreeSticker,
               let def = viewModel.stickerDefinition(for: selected.stickerId) {
                // Contextual inspector bar for the selected Free DIY sticker
                HStack(spacing: 10) {
                    Text(def.nameZh)
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(Color(red: 1.0, green: 0.86, blue: 0.42))
                        .lineLimit(1)

                    Spacer()

                    toolButton(icon: "arrow.left.and.right.righttriangle.left.righttriangle.right", label: "翻转") {
                        viewModel.updateSelectedFreeSticker { $0.isFlippedHorizontally.toggle() }
                    }
                    toolButton(icon: "square.3.layers.3d.top.filled", label: "置顶") {
                        viewModel.bringSelectedForward()
                    }
                    toolButton(icon: "square.3.layers.3d.bottom.filled", label: "置底") {
                        viewModel.sendSelectedBackward()
                    }
                    toolButton(icon: "plus.square.on.square", label: "复制") {
                        viewModel.duplicateSelectedFreeSticker()
                    }
                    toolButton(icon: "trash", label: "删除", tint: .red.opacity(0.9)) {
                        viewModel.deleteSelectedFreeSticker()
                    }
                }

                HStack(spacing: 14) {
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                            .font(.system(size: 11))
                            .foregroundStyle(.white.opacity(0.7))
                        Slider(
                            value: Binding(
                                get: { Double(selected.scale) },
                                set: { val in viewModel.updateSelectedFreeSticker { $0.scale = CGFloat(val) } }
                            ),
                            in: 0.5...1.8
                        )
                        .tint(Color(red: 1.0, green: 0.84, blue: 0.42))
                    }

                    HStack(spacing: 6) {
                        Image(systemName: "rotate.right")
                            .font(.system(size: 11))
                            .foregroundStyle(.white.opacity(0.7))
                        Slider(
                            value: Binding(
                                get: { selected.rotationDegrees },
                                set: { val in viewModel.updateSelectedFreeSticker { $0.rotationDegrees = val } }
                            ),
                            in: -45...45
                        )
                        .tint(Color(red: 0.55, green: 0.85, blue: 1.0))
                    }
                }
            } else {
                // General Free Mode actions when no sticker is selected
                HStack(spacing: 10) {
                    Button {
                        viewModel.loadOfficialLayoutIntoFreeMode()
                    } label: {
                        Label("载入官方参考", systemImage: "sparkles")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .background(Capsule().fill(Color.white.opacity(0.12)))
                    }

                    Spacer()

                    Button(action: onExportPhoto) {
                        Label("保存造景图", systemImage: "square.and.arrow.down")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(Color(red: 0.14, green: 0.12, blue: 0.20))
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(Capsule().fill(Color(red: 1.0, green: 0.84, blue: 0.42)))
                    }
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color(red: 0.15, green: 0.17, blue: 0.28).opacity(0.95))
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
                )
        )
        .padding(.horizontal, 10)
    }

    private func toolButton(icon: String, label: String, tint: Color = .white, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 2) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .semibold))
                Text(label)
                    .font(.system(size: 9.5))
            }
            .foregroundStyle(tint)
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color.white.opacity(0.08)))
        }
        .buttonStyle(.plain)
    }
}
