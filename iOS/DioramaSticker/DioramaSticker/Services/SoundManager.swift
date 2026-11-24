import AVFoundation

/// Procedural cozy sound effects synthesizer using AVAudioEngine (requires zero external audio files)
final class SoundManager {
    static let shared = SoundManager()

    var isMuted: Bool = false

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let sampleRate: Double = 44100.0
    private let format: AVAudioFormat

    private init() {
        format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        engine.mainMixerNode.outputVolume = 0.35
        try? AVAudioSession.sharedInstance().setCategory(.ambient, options: [.mixWithOthers])
    }

    private func ensureRunning() {
        guard !isMuted else { return }
        if !engine.isRunning {
            try? engine.start()
        }
        if !player.isPlaying {
            player.play()
        }
    }

    func playPickup() {
        playTone(frequencies: [440, 587], duration: 0.08, sweep: true)
    }

    func playSnap() {
        playTone(frequencies: [523.25, 659.25, 783.99], duration: 0.16, sweep: false)
    }

    func playReturn() {
        playTone(frequencies: [330, 261.63], duration: 0.10, sweep: true)
    }

    func playCompleteFanfare() {
        let notes: [Double] = [523.25, 659.25, 783.99, 1046.50]
        for (idx, freq) in notes.enumerated() {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(idx) * 0.11) {
                self.playTone(frequencies: [freq], duration: 0.28, sweep: false)
            }
        }
    }

    private func playTone(frequencies: [Double], duration: Double, sweep: Bool) {
        guard !isMuted else { return }
        ensureRunning()

        let frameCount = AVAudioFrameCount(sampleRate * duration)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return }
        buffer.frameLength = frameCount

        guard let samples = buffer.floatChannelData?[0] else { return }
        let total = Int(frameCount)

        for i in 0..<total {
            let t = Double(i) / sampleRate
            let progress = Double(i) / Double(total)
            // Soft attack and exponential decay envelope for cozy xylophone/pop timbre
            let attack = min(1.0, progress / 0.08)
            let decay = exp(-progress * 5.2)
            let env = Float(attack * decay)

            var value: Double = 0
            if sweep, frequencies.count >= 2 {
                let f = frequencies[0] + (frequencies[1] - frequencies[0]) * progress
                value = sin(2.0 * .pi * f * t)
            } else {
                for freq in frequencies {
                    value += sin(2.0 * .pi * freq * t) / Double(frequencies.count)
                }
            }
            samples[i] = Float(value) * env * 0.65
        }

        player.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
    }
}
