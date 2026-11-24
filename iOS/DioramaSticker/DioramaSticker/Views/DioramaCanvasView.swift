import SwiftUI

struct DioramaCanvasView: View {
    @ObservedObject var viewModel: DioramaGameViewModel

    @GestureState private var gestureMagnify: CGFloat = 1.0
    @GestureState private var gesturePan: CGSize = .zero

    var body: some View {
        GeometryReader { outerGeo in
            let aspect = viewModel.level?.aspectRatio ?? (896.0 / 1200.0)
            let fitSize = aspectFitSize(in: outerGeo.size, aspect: aspect)

            ZStack {
                canvasContent(size: fitSize)
                    .frame(width: fitSize.width, height: fitSize.height)
                    .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                    .shadow(color: .black.opacity(0.42), radius: 18, x: 0, y: 10)
                    .background(
                        GeometryReader { canvasGeo in
                            Color.clear
                                .onAppear {
                                    viewModel.canvasGlobalFrame = canvasGeo.frame(in: .global)
                                }
                                .onChange(of: canvasGeo.frame(in: .global)) { _, newFrame in
                                    viewModel.canvasGlobalFrame = newFrame
                                }
                        }
                    )
                    .scaleEffect(clampedZoom)
                    .offset(
                        x: viewModel.panOffset.width + gesturePan.width,
                        y: viewModel.panOffset.height + gesturePan.height
                    )
                    .gesture(magnificationGesture)
                    .simultaneousGesture(panGesture)
                    .onTapGesture {
                        if viewModel.gameMode == .freeDIY {
                            viewModel.selectedFreeInstanceID = nil
                        }
                    }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var clampedZoom: CGFloat {
        min(max(viewModel.zoomScale * gestureMagnify, 1.0), 2.8)
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .updating($gestureMagnify) { value, state, _ in
                state = value
            }
            .onEnded { value in
                withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                    viewModel.zoomScale = min(max(viewModel.zoomScale * value, 1.0), 2.8)
                    if viewModel.zoomScale <= 1.02 {
                        viewModel.zoomScale = 1.0
                        viewModel.panOffset = .zero
                    }
                }
            }
    }

    private var panGesture: some Gesture {
        DragGesture(minimumDistance: 12)
            .updating($gesturePan) { value, state, _ in
                guard viewModel.zoomScale > 1.05, viewModel.activeDrag == nil else { return }
                state = value.translation
            }
            .onEnded { value in
                guard viewModel.zoomScale > 1.05, viewModel.activeDrag == nil else { return }
                let maxX = viewModel.canvasGlobalFrame.width * (viewModel.zoomScale - 1) * 0.55
                let maxY = viewModel.canvasGlobalFrame.height * (viewModel.zoomScale - 1) * 0.55
                withAnimation(.spring(response: 0.25, dampingFraction: 0.85)) {
                    viewModel.panOffset = CGSize(
                        width: min(max(viewModel.panOffset.width + value.translation.width, -maxX), maxX),
                        height: min(max(viewModel.panOffset.height + value.translation.height, -maxY), maxY)
                    )
                }
            }
    }

    @ViewBuilder
    func canvasContent(size: CGSize) -> some View {
        let folder = viewModel.currentFolderName
        ZStack {
            // 1. Empty diorama base background
            if let bgName = viewModel.level?.backgroundFile,
               let bgImg = LevelLoader.shared.loadImage(named: bgName, folderName: folder) {
                Image(uiImage: bgImg)
                    .resizable()
                    .scaledToFill()
                    .frame(width: size.width, height: size.height)
            }

            // 2. Classic Mode: subtle silhouette outlines & active magnetic target pulse
            if viewModel.gameMode == .classic, let stickers = viewModel.level?.stickers {
                if viewModel.showOutlineHints {
                    ForEach(stickers.filter { !viewModel.placedStickerIDs.contains($0.id) }) { sticker in
                        if let img = LevelLoader.shared.loadImage(named: sticker.file, folderName: folder) {
                            Image(uiImage: img)
                                .resizable()
                                .renderingMode(.template)
                                .foregroundStyle(Color.yellow.opacity(0.20))
                                .frame(width: sticker.normW * size.width, height: sticker.normH * size.height)
                                .position(x: sticker.normX * size.width, y: sticker.normY * size.height)
                                .allowsHitTesting(false)
                        }
                    }
                }

                // Magnetic target ring when dragging near correct slot or using Spotlight Hint
                if let targetSticker = highlightedTargetSticker {
                    MagneticTargetRing(
                        width: max(48, targetSticker.normW * size.width),
                        height: max(48, targetSticker.normH * size.height),
                        isLockedOn: viewModel.activeDrag?.isNearTarget == true
                    )
                    .position(x: targetSticker.normX * size.width, y: targetSticker.normY * size.height)
                    .allowsHitTesting(false)
                }

                // Placed classic stickers in zIndex order
                ForEach(viewModel.placedClassicStickers) { sticker in
                    if let img = LevelLoader.shared.loadImage(named: sticker.file, folderName: folder) {
                        Image(uiImage: img)
                            .resizable()
                            .frame(width: sticker.normW * size.width, height: sticker.normH * size.height)
                            .position(x: sticker.normX * size.width, y: sticker.normY * size.height)
                            .transition(.scale(scale: 1.18).combined(with: .opacity))
                            .allowsHitTesting(false)
                    }
                }
            }

            // 3. Free DIY Mode: draggable, scalable, rotatable stickers
            if viewModel.gameMode == .freeDIY {
                ForEach(viewModel.freePlacedStickers.sorted { $0.zIndex < $1.zIndex }) { item in
                    if let def = viewModel.stickerDefinition(for: item.stickerId),
                       let img = LevelLoader.shared.loadImage(named: def.file, folderName: folder) {
                        let isSelected = viewModel.selectedFreeInstanceID == item.instanceId
                        let isBeingDragged = viewModel.activeDrag?.freeInstanceId == item.instanceId

                        Image(uiImage: img)
                            .resizable()
                            .scaleEffect(x: item.isFlippedHorizontally ? -1 : 1, y: 1)
                            .frame(width: def.normW * size.width * item.scale,
                                   height: def.normH * size.height * item.scale)
                            .overlay {
                                if isSelected {
                                    RoundedRectangle(cornerRadius: 10)
                                        .strokeBorder(Color(red: 1.0, green: 0.85, blue: 0.35),
                                                      style: StrokeStyle(lineWidth: 2, dash: [5, 4]))
                                        .padding(-5)
                                }
                            }
                            .rotationEffect(.degrees(item.rotationDegrees))
                            .opacity(isBeingDragged ? 0.25 : 1.0)
                            .position(x: item.normX * size.width, y: item.normY * size.height)
                            .onTapGesture {
                                withAnimation(.easeInOut(duration: 0.15)) {
                                    viewModel.selectedFreeInstanceID = item.instanceId
                                }
                                HapticManager.shared.magneticEnter()
                            }
                            .gesture(
                                DragGesture(coordinateSpace: .global)
                                    .onChanged { value in
                                        if viewModel.activeDrag == nil {
                                            viewModel.beginDrag(sticker: def, at: value.location, freeInstanceId: item.instanceId)
                                        } else {
                                            viewModel.updateDrag(to: value.location)
                                        }
                                    }
                                    .onEnded { value in
                                        viewModel.endDrag(at: value.location)
                                    }
                            )
                    }
                }
            }

            // 4. Peek Reference Complete Image Overlay
            if viewModel.isPeekingReference,
               let refImg = LevelLoader.shared.loadImage(named: "reference_complete.png", folderName: folder) {
                Image(uiImage: refImg)
                    .resizable()
                    .scaledToFill()
                    .frame(width: size.width, height: size.height)
                    .opacity(0.76)
                    .transition(.opacity)
                    .allowsHitTesting(false)
            }

            // 5. Sparkle burst particles on snap
            ForEach(viewModel.sparkles) { burst in
                SparkleBurstView()
                    .position(x: burst.normX * size.width, y: burst.normY * size.height)
                    .allowsHitTesting(false)
            }
        }
    }

    private var highlightedTargetSticker: StickerDefinition? {
        if let drag = viewModel.activeDrag, drag.isNearTarget {
            return drag.sticker
        }
        if let hintID = viewModel.spotlightStickerID {
            return viewModel.stickerDefinition(for: hintID)
        }
        return nil
    }

    private func aspectFitSize(in container: CGSize, aspect: CGFloat) -> CGSize {
        let maxW = max(1, container.width - 16)
        let maxH = max(1, container.height - 8)
        if maxW / maxH > aspect {
            return CGSize(width: maxH * aspect, height: maxH)
        } else {
            return CGSize(width: maxW, height: maxW / aspect)
        }
    }
}

/// Pulsing golden ring shown when a sticker enters the magnetic snap radius or spotlight hint
private struct MagneticTargetRing: View {
    let width: CGFloat
    let height: CGFloat
    let isLockedOn: Bool
    @State private var pulse = false

    var body: some View {
        Ellipse()
            .strokeBorder(
                isLockedOn ? Color(red: 1.0, green: 0.88, blue: 0.35) : Color.white.opacity(0.85),
                style: StrokeStyle(lineWidth: isLockedOn ? 3.5 : 2.5, dash: isLockedOn ? [] : [7, 5])
            )
            .background(
                Ellipse().fill(
                    (isLockedOn ? Color.yellow : Color.white).opacity(isLockedOn ? 0.22 : 0.12)
                )
            )
            .frame(width: width * 1.08, height: height * 1.08)
            .scaleEffect(pulse ? 1.06 : 0.96)
            .shadow(color: .yellow.opacity(0.65), radius: 12)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.65).repeatForever(autoreverses: true)) {
                    pulse = true
                }
            }
    }
}

/// Star/firefly burst particle animation triggered on sticker placement
private struct SparkleBurstView: View {
    @State private var expanded = false

    var body: some View {
        ZStack {
            ForEach(0..<10, id: \.self) { idx in
                let angle = Double(idx) * (360.0 / 10.0) * .pi / 180.0
                Image(systemName: idx % 2 == 0 ? "sparkle" : "star.fill")
                    .font(.system(size: idx % 2 == 0 ? 15 : 10, weight: .bold))
                    .foregroundStyle(idx % 2 == 0 ? Color(red: 1.0, green: 0.90, blue: 0.40) : Color.white)
                    .offset(
                        x: expanded ? CGFloat(cos(angle)) * 46 : 0,
                        y: expanded ? CGFloat(sin(angle)) * 46 : 0
                    )
                    .scaleEffect(expanded ? 0.35 : 1.15)
                    .opacity(expanded ? 0 : 1)
            }
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.72)) {
                expanded = true
            }
        }
    }
}
