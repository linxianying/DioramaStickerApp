import SwiftUI

@main
struct DioramaStickerApp: App {
    @StateObject private var viewModel = DioramaGameViewModel()

    var body: some Scene {
        WindowGroup {
            LevelGalleryView(viewModel: viewModel)
                .preferredColorScheme(.dark)
        }
    }
}
