import SwiftUI

struct LevelGalleryView: View {
    @ObservedObject var viewModel: DioramaGameViewModel
    @State private var activeLevelItem: LevelManifestItem? = nil
    @State private var showSettings = false

    private let columns = [
        GridItem(.flexible(), spacing: 14),
        GridItem(.flexible(), spacing: 14)
    ]

    var body: some View {
        NavigationStack {
            ZStack {
                LinearGradient(
                    colors: [
                        Color(red: 0.09, green: 0.11, blue: 0.20),
                        Color(red: 0.14, green: 0.16, blue: 0.28),
                        Color(red: 0.08, green: 0.10, blue: 0.18)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                ScrollView(showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 18) {
                        headerSection

                        LazyVGrid(columns: columns, spacing: 14) {
                            ForEach(viewModel.manifest) { item in
                                LevelCard(
                                    item: item,
                                    placed: viewModel.savedPlacedCount(for: item.id),
                                    total: viewModel.totalStickerCount(for: item)
                                ) {
                                    HapticManager.shared.stickerPickup()
                                    viewModel.selectLevel(item)
                                    activeLevelItem = item
                                }
                            }
                        }

                        footerTip
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 14)
                    .padding(.bottom, 28)
                }
            }
            .navigationDestination(item: $activeLevelItem) { _ in
                MainGameView(viewModel: viewModel)
                    .navigationBarBackButtonHidden(true)
            }
            .sheet(isPresented: $showSettings) {
                SettingsSheetView(viewModel: viewModel)
                    .presentationDetents([.medium, .large])
                    .presentationDragIndicator(.visible)
            }
        }
    }

    private var headerSection: some View {
        HStack(alignment: .center, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text("微缩造景贴纸屋")
                    .font(.system(size: 25, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                Text("挑一个治愈小世界，把贴纸一个个贴回原位 ✨")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.white.opacity(0.72))
            }
            Spacer()
            let totalDone = viewModel.manifest.reduce(0) { $0 + viewModel.savedPlacedCount(for: $1.id) }
            let totalAll = viewModel.manifest.reduce(0) { $0 + viewModel.totalStickerCount(for: $1) }
            VStack(spacing: 2) {
                Text("\(totalDone)/\(totalAll)")
                    .font(.system(size: 13, weight: .bold, design: .monospaced))
                    .foregroundStyle(Color(red: 1.0, green: 0.85, blue: 0.42))
                Text("已收集")
                    .font(.system(size: 10))
                    .foregroundStyle(.white.opacity(0.65))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.white.opacity(0.09))
                    .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(Color.white.opacity(0.14), lineWidth: 1))
            )

            Button {
                HapticManager.shared.magneticEnter()
                showSettings = true
            } label: {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 38, height: 38)
                    .background(
                        Circle()
                            .fill(Color.white.opacity(0.11))
                            .overlay(Circle().strokeBorder(Color.white.opacity(0.16), lineWidth: 1))
                    )
            }
            .buttonStyle(.plain)
        }
    }

    private var footerTip: some View {
        HStack(spacing: 10) {
            Image(systemName: "wand.and.stars")
                .foregroundStyle(Color(red: 1.0, green: 0.84, blue: 0.42))
            Text("每个主题均支持「闯关找位」与「自由造景 DIY」，并可保存高清作品到相册。")
                .font(.system(size: 12))
                .foregroundStyle(.white.opacity(0.75))
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color.white.opacity(0.07))
        )
    }
}

private struct LevelCard: View {
    let item: LevelManifestItem
    let placed: Int
    let total: Int
    let onTap: () -> Void

    private var isComplete: Bool {
        total > 0 && placed >= total
    }

    private var fraction: Double {
        guard total > 0 else { return 0 }
        return Double(placed) / Double(total)
    }

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                ZStack(alignment: .topTrailing) {
                    if let img = LevelLoader.shared.loadImage(named: "reference_complete.png", folderName: item.folder) {
                        Image(uiImage: img)
                            .resizable()
                            .scaledToFill()
                            .frame(height: 176)
                            .clipped()
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    } else {
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color.white.opacity(0.08))
                            .frame(height: 176)
                    }

                    Text(isComplete ? "已通关 ✓" : "\(placed)/\(total)")
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .foregroundStyle(isComplete ? Color(red: 0.14, green: 0.12, blue: 0.20) : .white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            Capsule().fill(
                                isComplete
                                    ? Color(red: 1.0, green: 0.84, blue: 0.42)
                                    : Color.black.opacity(0.55)
                            )
                        )
                        .padding(8)
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text("\(item.emoji) \(item.titleZh)")
                        .font(.system(size: 14.5, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                        .lineLimit(1)

                    Text(item.subtitleZh)
                        .font(.system(size: 11.5))
                        .foregroundStyle(.white.opacity(0.68))
                        .lineLimit(1)

                    ProgressView(value: fraction)
                        .tint(Color(red: 1.0, green: 0.84, blue: 0.42))
                        .padding(.top, 3)
                }
                .padding(.horizontal, 4)
                .padding(.bottom, 4)
            }
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(Color(red: 0.16, green: 0.19, blue: 0.31).opacity(0.92))
                    .overlay(
                        RoundedRectangle(cornerRadius: 22, style: .continuous)
                            .strokeBorder(
                                isComplete ? Color(red: 1.0, green: 0.84, blue: 0.42).opacity(0.55) : Color.white.opacity(0.12),
                                lineWidth: 1.2
                            )
                    )
            )
            .shadow(color: .black.opacity(0.32), radius: 10, y: 6)
        }
        .buttonStyle(.plain)
    }
}
