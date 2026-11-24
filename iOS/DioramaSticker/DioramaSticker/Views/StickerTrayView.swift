import SwiftUI

struct StickerTrayView: View {
    @ObservedObject var viewModel: DioramaGameViewModel

    var body: some View {
        VStack(spacing: 10) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(viewModel.availableCategories) { cat in
                        let isSelected = viewModel.selectedCategory == cat
                        let count = remainingCount(for: cat)

                        Button {
                            withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                                viewModel.selectedCategory = cat
                            }
                            HapticManager.shared.magneticEnter()
                        } label: {
                            HStack(spacing: 4) {
                                Text(cat.icon)
                                    .font(.system(size: 13))
                                Text(cat.displayLabel)
                                    .font(.system(size: 12, weight: .semibold))
                                if viewModel.gameMode == .classic {
                                    Text("\(count)")
                                        .font(.system(size: 11, weight: .bold, design: .rounded))
                                        .padding(.horizontal, 5)
                                        .padding(.vertical, 1)
                                        .background(
                                            Capsule().fill(isSelected ? Color.black.opacity(0.22) : Color.white.opacity(0.12))
                                        )
                                }
                            }
                            .foregroundStyle(isSelected ? Color(red: 0.14, green: 0.12, blue: 0.20) : Color.white.opacity(0.88))
                            .padding(.horizontal, 11)
                            .padding(.vertical, 6)
                            .background(
                                Capsule().fill(
                                    isSelected
                                        ? Color(red: 1.0, green: 0.84, blue: 0.42)
                                        : Color.white.opacity(0.10)
                                )
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 14)
            }

            if viewModel.trayStickers.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.seal.fill")
                        .foregroundStyle(Color(red: 1.0, green: 0.84, blue: 0.42))
                    Text("这一类贴纸全部贴完啦！")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(.white.opacity(0.82))
                }
                .frame(height: 92)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    LazyHStack(spacing: 12) {
                        ForEach(viewModel.trayStickers) { sticker in
                            StickerCardCell(
                                sticker: sticker,
                                folderName: viewModel.currentFolderName,
                                isBeingDragged: viewModel.activeDrag?.sticker.id == sticker.id && viewModel.activeDrag?.freeInstanceId == nil,
                                isHinted: viewModel.spotlightStickerID == sticker.id,
                                onDragChanged: { loc in
                                    if viewModel.activeDrag == nil {
                                        viewModel.beginDrag(sticker: sticker, at: loc)
                                    } else {
                                        viewModel.updateDrag(to: loc)
                                    }
                                },
                                onDragEnded: { loc in
                                    viewModel.endDrag(at: loc)
                                }
                            )
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 4)
                }
                .frame(height: 98)
            }
        }
        .padding(.top, 10)
        .padding(.bottom, 10)
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(Color(red: 0.13, green: 0.15, blue: 0.25).opacity(0.95))
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .strokeBorder(Color.white.opacity(0.12), lineWidth: 1)
                )
        )
        .padding(.horizontal, 10)
    }

    private func remainingCount(for category: StickerCategory) -> Int {
        guard let all = viewModel.level?.stickers else { return 0 }
        let unplaced = all.filter { !viewModel.placedStickerIDs.contains($0.id) }
        if category == .all { return unplaced.count }
        return unplaced.filter { $0.category == category.rawValue }.count
    }
}

private struct StickerCardCell: View {
    let sticker: StickerDefinition
    let folderName: String
    let isBeingDragged: Bool
    let isHinted: Bool
    let onDragChanged: (CGPoint) -> Void
    let onDragEnded: (CGPoint) -> Void

    var body: some View {
        VStack(spacing: 4) {
            ZStack {
                if let img = LevelLoader.shared.loadImage(named: sticker.file, folderName: folderName) {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .shadow(color: .white.opacity(0.9), radius: 1.2)
                        .shadow(color: .black.opacity(0.35), radius: 4, x: 0, y: 3)
                        .padding(6)
                }
            }
            .frame(width: 66, height: 64)

            Text(sticker.nameZh)
                .font(.system(size: 10.5, weight: .medium))
                .foregroundStyle(.white.opacity(0.88))
                .lineLimit(1)
        }
        .frame(width: 78, height: 88)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.white.opacity(isHinted ? 0.18 : 0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(
                            isHinted ? Color(red: 1.0, green: 0.85, blue: 0.35) : Color.white.opacity(0.12),
                            lineWidth: isHinted ? 2 : 1
                        )
                )
        )
        .opacity(isBeingDragged ? 0.28 : 1.0)
        .scaleEffect(isBeingDragged ? 0.9 : 1.0)
        .gesture(
            DragGesture(minimumDistance: 6, coordinateSpace: .global)
                .onChanged { value in onDragChanged(value.location) }
                .onEnded { value in onDragEnded(value.location) }
        )
    }
}
