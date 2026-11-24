import SwiftUI
import UIKit

struct MainGameView: View {
    @ObservedObject var viewModel: DioramaGameViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var showResetConfirm = false
    @State private var showSettings = false

    var body: some View {
        ZStack {
            // Cozy starry night gradient backdrop
            LinearGradient(
                colors: [
                    Color(red: 0.09, green: 0.11, blue: 0.20),
                    Color(red: 0.13, green: 0.15, blue: 0.26),
                    Color(red: 0.08, green: 0.10, blue: 0.18)
                ],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 8) {
                topHeaderBar
                actionStrip
                DioramaCanvasView(viewModel: viewModel)
                    .layoutPriority(1)

                if viewModel.gameMode == .freeDIY {
                    FreeModeToolbarView(viewModel: viewModel, onExportPhoto: exportCustomDiorama)
                }

                StickerTrayView(viewModel: viewModel)
            }
            .padding(.top, 4)

            // Floating lifted sticker overlay during active drag (+ optional 3D peel & tweezers!)
            if let drag = viewModel.activeDrag,
               let img = LevelLoader.shared.loadImage(named: drag.sticker.file, folderName: viewModel.currentFolderName) {
                let targetSize = floatingStickerSize(for: drag.sticker)
                let offsetY = (drag.freeInstanceId == nil) ? viewModel.dragFingerOffsetY : 0

                ZStack {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .frame(width: targetSize.width, height: targetSize.height)
                        .modifier(StickerPeelEffectModifier(enabled: viewModel.enablePeelAnimation && drag.freeInstanceId == nil))
                        .shadow(color: .white.opacity(drag.isNearTarget ? 1.0 : 0.9), radius: 2.2)
                        .shadow(color: drag.isNearTarget ? .yellow.opacity(0.75) : .black.opacity(0.45),
                                radius: drag.isNearTarget ? 14 : 12,
                                x: 0, y: 12)
                        .scaleEffect(drag.isNearTarget ? 1.04 : 1.08)
                        .rotationEffect(.degrees(drag.isNearTarget ? 0 : -4))

                    if viewModel.pickupToolStyle == .tweezers {
                        TweezerOverlayView(
                            theme: viewModel.tweezerTheme,
                            isPinching: true,
                            isNearTarget: drag.isNearTarget
                        )
                        .offset(x: targetSize.width * 0.22 + 34, y: -targetSize.height * 0.18 - 34)
                    }
                }
                .id(drag.sticker.id)
                .position(
                    x: drag.globalLocation.x,
                    y: drag.globalLocation.y + offsetY
                )
                .ignoresSafeArea()
                .allowsHitTesting(false)
                .animation(.interactiveSpring(response: 0.18, dampingFraction: 0.72), value: drag.isNearTarget)
            }

            // Toast notification banner
            if let toast = viewModel.toastMessage {
                VStack {
                    Text(toast)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 9)
                        .background(
                            Capsule()
                                .fill(Color(red: 0.16, green: 0.19, blue: 0.32).opacity(0.96))
                                .overlay(Capsule().strokeBorder(Color(red: 1.0, green: 0.85, blue: 0.40).opacity(0.5), lineWidth: 1))
                        )
                        .shadow(color: .black.opacity(0.35), radius: 10, y: 4)
                        .padding(.top, 60)
                    Spacer()
                }
                .transition(.move(edge: .top).combined(with: .opacity))
                .allowsHitTesting(false)
            }

            // Classic Mode Completion Celebration Modal
            if viewModel.showCompletionBanner {
                completionOverlay
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsSheetView(viewModel: viewModel)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .confirmationDialog("确定要重置当前画布吗？", isPresented: $showResetConfirm, titleVisibility: .visible) {
            Button("重置", role: .destructive) {
                viewModel.resetCurrentMode()
            }
            Button("取消", role: .cancel) {}
        }
    }

    // MARK: - Top Header Bar
    private var topHeaderBar: some View {
        HStack(spacing: 8) {
            Button {
                HapticManager.shared.magneticEnter()
                dismiss()
            } label: {
                Image(systemName: "chevron.left")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 32, height: 32)
                    .background(Circle().fill(Color.white.opacity(0.12)))
            }
            .buttonStyle(.plain)

            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.level?.titleZh ?? "造景贴纸")
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .lineLimit(1)

                if viewModel.gameMode == .classic {
                    HStack(spacing: 6) {
                        ProgressView(value: viewModel.progressFraction)
                            .tint(Color(red: 1.0, green: 0.84, blue: 0.42))
                            .frame(width: 72)
                        Text("\(viewModel.placedCount)/\(viewModel.totalStickerCount)")
                            .font(.system(size: 11.5, weight: .bold, design: .monospaced))
                            .foregroundStyle(Color(red: 1.0, green: 0.88, blue: 0.50))
                    }
                } else {
                    Text("已摆放 \(viewModel.freePlacedStickers.count) 张贴纸")
                        .font(.system(size: 11.5, weight: .medium))
                        .foregroundStyle(Color(red: 0.72, green: 0.86, blue: 1.0))
                }
            }

            Spacer()

            // Dual Mode Segmented Toggle
            HStack(spacing: 2) {
                ForEach(GameMode.allCases) { mode in
                    let active = viewModel.gameMode == mode
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.78)) {
                            viewModel.gameMode = mode
                            viewModel.selectedFreeInstanceID = nil
                        }
                        HapticManager.shared.magneticEnter()
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: mode.systemImage)
                                .font(.system(size: 11, weight: .semibold))
                            Text(mode.title)
                                .font(.system(size: 12, weight: .bold))
                        }
                        .foregroundStyle(active ? Color(red: 0.14, green: 0.12, blue: 0.20) : .white.opacity(0.75))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            Capsule().fill(active ? Color(red: 1.0, green: 0.84, blue: 0.42) : Color.clear)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(3)
            .background(Capsule().fill(Color.white.opacity(0.10)))
        }
        .padding(.horizontal, 14)
    }

    // MARK: - Utility Strip (Hints, Peek Reference, Zoom Reset, Settings, Reset)
    private var actionStrip: some View {
        HStack(spacing: 8) {
            if viewModel.gameMode == .classic {
                pillButton(
                    icon: viewModel.showOutlineHints ? "eye.fill" : "eye.slash",
                    title: "轮廓",
                    active: viewModel.showOutlineHints
                ) {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        viewModel.showOutlineHints.toggle()
                    }
                }

                // Hold to Peek Reference Image
                HStack(spacing: 4) {
                    Image(systemName: "photo.on.rectangle")
                        .font(.system(size: 12, weight: .semibold))
                    Text("按住看原图")
                        .font(.system(size: 11.5, weight: .semibold))
                }
                .foregroundStyle(viewModel.isPeekingReference ? Color(red: 0.14, green: 0.12, blue: 0.20) : .white.opacity(0.9))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    Capsule().fill(viewModel.isPeekingReference ? Color(red: 1.0, green: 0.84, blue: 0.42) : Color.white.opacity(0.11))
                )
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { _ in
                            if !viewModel.isPeekingReference {
                                withAnimation(.easeInOut(duration: 0.18)) {
                                    viewModel.isPeekingReference = true
                                }
                                HapticManager.shared.magneticEnter()
                            }
                        }
                        .onEnded { _ in
                            withAnimation(.easeInOut(duration: 0.18)) {
                                viewModel.isPeekingReference = false
                            }
                        }
                )

                pillButton(icon: "lightbulb.fill", title: "提示", active: viewModel.spotlightStickerID != nil) {
                    viewModel.triggerSpotlightHint()
                }
            }

            Spacer()

            if viewModel.zoomScale > 1.05 {
                pillButton(icon: "arrow.down.right.and.arrow.up.left", title: "复原", active: true) {
                    viewModel.resetCamera()
                }
            }

            iconCircleButton(icon: "gearshape.fill") {
                HapticManager.shared.magneticEnter()
                showSettings = true
            }

            iconCircleButton(icon: "arrow.counterclockwise") {
                showResetConfirm = true
            }
        }
        .padding(.horizontal, 14)
    }

    private func pillButton(icon: String, title: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 11.5, weight: .semibold))
                Text(title)
                    .font(.system(size: 11.5, weight: .semibold))
            }
            .foregroundStyle(active ? Color(red: 0.14, green: 0.12, blue: 0.20) : .white.opacity(0.88))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule().fill(active ? Color(red: 1.0, green: 0.84, blue: 0.42) : Color.white.opacity(0.11))
            )
        }
        .buttonStyle(.plain)
    }

    private func iconCircleButton(icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.white.opacity(0.88))
                .frame(width: 30, height: 30)
                .background(Circle().fill(Color.white.opacity(0.11)))
        }
        .buttonStyle(.plain)
    }

    private func floatingStickerSize(for sticker: StickerDefinition) -> CGSize {
        let baseW = viewModel.canvasGlobalFrame.width > 1 ? viewModel.canvasGlobalFrame.width : 330
        let baseH = viewModel.canvasGlobalFrame.height > 1 ? viewModel.canvasGlobalFrame.height : 440
        let w = max(46, sticker.normW * baseW * viewModel.zoomScale)
        let h = max(46, sticker.normH * baseH * viewModel.zoomScale)
        return CGSize(width: w, height: h)
    }

    // MARK: - Completion Celebration Overlay
    private var completionOverlay: some View {
        ZStack {
            Color.black.opacity(0.58)
                .ignoresSafeArea()
                .onTapGesture {
                    withAnimation { viewModel.showCompletionBanner = false }
                }

            VStack(spacing: 16) {
                Text("🎉")
                    .font(.system(size: 48))

                Text("\(viewModel.level?.titleZh ?? "造景")完成！")
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)

                Text("全部 \(viewModel.totalStickerCount) 张贴纸已完美归位\n现已解锁「自由造景模式」，随心搭配你的小世界！")
                    .font(.system(size: 14))
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white.opacity(0.82))

                HStack(spacing: 12) {
                    Button {
                        withAnimation { viewModel.showCompletionBanner = false }
                    } label: {
                        Text("欣赏作品")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 18)
                            .padding(.vertical, 11)
                            .background(Capsule().fill(Color.white.opacity(0.15)))
                    }

                    Button {
                        withAnimation {
                            viewModel.showCompletionBanner = false
                            viewModel.gameMode = .freeDIY
                            if viewModel.freePlacedStickers.isEmpty {
                                viewModel.loadOfficialLayoutIntoFreeMode()
                            }
                        }
                    } label: {
                        Text("进入自由造景 ✨")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(Color(red: 0.14, green: 0.12, blue: 0.20))
                            .padding(.horizontal, 18)
                            .padding(.vertical, 11)
                            .background(Capsule().fill(Color(red: 1.0, green: 0.84, blue: 0.42)))
                    }
                }
                .padding(.top, 6)
            }
            .padding(26)
            .background(
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .fill(Color(red: 0.14, green: 0.16, blue: 0.28))
                    .overlay(
                        RoundedRectangle(cornerRadius: 28, style: .continuous)
                            .strokeBorder(Color(red: 1.0, green: 0.85, blue: 0.40).opacity(0.55), lineWidth: 1.5)
                    )
            )
            .padding(.horizontal, 32)
            .shadow(color: .black.opacity(0.5), radius: 24, y: 12)
            .transition(.scale(scale: 0.86).combined(with: .opacity))
        }
    }

    // MARK: - Export Custom Diorama Poster to iOS Photos
    private func exportCustomDiorama() {
        let prevSelection = viewModel.selectedFreeInstanceID
        viewModel.selectedFreeInstanceID = nil

        let exportSize = CGSize(width: 896, height: 1200)
        let renderer = ImageRenderer(
            content: DioramaCanvasView(viewModel: viewModel)
                .canvasContent(size: exportSize)
                .frame(width: exportSize.width, height: exportSize.height)
        )
        renderer.scale = 1.0

        if let uiImage = renderer.uiImage {
            UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
            HapticManager.shared.levelComplete()
            viewModel.showToast("已保存高清造景海报到系统相册 📸")
        } else {
            viewModel.showToast("导出失败，请重试")
        }
        viewModel.selectedFreeInstanceID = prevSelection
    }
}
