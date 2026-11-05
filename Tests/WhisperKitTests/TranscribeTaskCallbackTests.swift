import CoreML
import XCTest
@testable import WhisperKit

final class TranscribeTaskCallbackTests: XCTestCase {
    func testAbsentCallbackStaysAbsentForEveryAttemptAndWindow() async throws {
        for windows in [1, 2] {
            for attempts in [1, 6] {
                let fixture = try Fixture(windows: windows, attempts: attempts)
                let result = try await fixture.run()

                XCTAssertEqual(fixture.decoder.callbackPresence, Array(repeating: false, count: windows * attempts))
                assertCompleted(fixture, result: result, windows: windows, attempts: attempts)
            }
        }
    }

    func testNoConsumerAndContinuingConsumerHaveIdenticalSelectedOutputAndWork() async throws {
        let absent = try Fixture(windows: 2, attempts: 6)
        let present = try Fixture(windows: 2, attempts: 6)
        let absentResult = try await absent.run()
        let presentResult = try await present.run(callback: { _ in nil })

        assertCompleted(absent, result: absentResult, windows: 2, attempts: 6)
        assertCompleted(present, result: presentResult, windows: 2, attempts: 6)
        XCTAssertEqual(present.decoder.callbackPresence, Array(repeating: true, count: 12))
        XCTAssertEqual(absentResult.text, presentResult.text)
        XCTAssertEqual(absentResult.language, presentResult.language)
        XCTAssertEqual(absentResult.segments, presentResult.segments)
        XCTAssertEqual(absent.decoder.inputs.resets, present.decoder.inputs.resets)
        XCTAssertEqual(absent.decoder.receivedOptions.count, present.decoder.receivedOptions.count)
    }

    func testConsumerReceivesExistingWindowMappingAndUnchangedProgress() async throws {
        let fixture = try Fixture(windows: 2, attempts: 6)
        var progress: [TranscriptionProgress] = []
        _ = try await fixture.run(callback: {
            progress.append($0)
            return nil
        })

        // The existing mapping subtracts the last zero-based fallback index.
        XCTAssertEqual(progress.map(\.windowId), [0, 0, -1, -2, -3, -4, -3, 1, 0, -1, -2, -3])
        XCTAssertEqual(progress.count, 12)
        for value in progress {
            XCTAssertEqual(value.text, "synthetic progress")
            XCTAssertEqual(value.tokens, [7, 8])
            XCTAssertEqual(value.temperature, 0.4)
            XCTAssertEqual(value.avgLogprob, -0.5)
            XCTAssertEqual(value.compressionRatio, 1.2)
            XCTAssertEqual(value.timings.totalDecodingLoops, 17)
        }
    }

    func testConsumerReturnValuesAreForwardedWithoutInterpretation() async throws {
        let decisions: [Bool?] = [nil, true, false]
        for decision in decisions {
            let fixture = try Fixture(windows: 1, attempts: 1)
            var calls = 0
            _ = try await fixture.run(callback: { _ in
                calls += 1
                return decision
            })
            XCTAssertEqual(calls, 1)
            XCTAssertEqual(fixture.decoder.callbackPresence, [true])
            XCTAssertEqual(fixture.decoder.decisions.count, 1)
            XCTAssertEqual(fixture.decoder.decisions[0], decision)
        }
    }

    func testRetainedConsumerWrapperDoesNotRetainTranscribeTask() async throws {
        let fixture = try Fixture(windows: 1, attempts: 1)
        var task: TranscribeTask? = fixture.makeTask()
        weak var weakTask = task
        var calls = 0
        _ = try await task?.run(audioArray: fixture.audio, decodeOptions: fixture.options, callback: { _ in
            calls += 1
            return false
        })
        let retainedCallback = try XCTUnwrap(fixture.decoder.retainedCallback)
        task = nil

        XCTAssertNil(weakTask)
        XCTAssertNil(retainedCallback(RecordingDecoder.progress()))
        XCTAssertEqual(calls, 1)
    }

    func testDecoderFailurePropagatesWithoutAnotherFallbackOrWindow() async throws {
        for hasConsumer in [false, true] {
            let fixture = try Fixture(windows: 2, attempts: 6)
            fixture.decoder.failAtInvocation = 2
            let callback: TranscriptionCallback = hasConsumer ? { _ in nil } : nil
            do {
                _ = try await fixture.run(callback: callback)
                XCTFail("Expected scripted decoder failure")
            } catch {
                XCTAssertEqual(error as? ScriptError, .decoderFailure)
            }
            XCTAssertEqual(fixture.decoder.callbackPresence, [hasConsumer, hasConsumer])
            XCTAssertEqual(fixture.encoder.calls, 1)
            XCTAssertEqual(fixture.decoder.inputs.resets.count, 1)
        }
    }

    func testCancellationStillReachesDecoderOnNextFallbackAttempt() async throws {
        for hasConsumer in [false, true] {
            let fixture = try Fixture(windows: 2, attempts: 6)
            fixture.decoder.cancelAfterFirstResult = true
            let callback: TranscriptionCallback = hasConsumer ? { _ in nil } : nil
            let operation = Task {
                try await fixture.run(callback: callback)
            }
            do {
                _ = try await operation.value
                XCTFail("Expected cancellation")
            } catch {
                XCTAssertTrue(error is CancellationError)
            }
            XCTAssertEqual(fixture.decoder.callbackPresence, [hasConsumer, hasConsumer])
            XCTAssertEqual(fixture.decoder.decisions.count, 1)
            XCTAssertEqual(fixture.encoder.calls, 1)
        }
    }

    private func assertCompleted(
        _ fixture: Fixture, result: TranscriptionResult, windows: Int, attempts: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(result.segments.map(\.tokens), (0..<windows).map { [40 + $0] }, file: file, line: line)
        XCTAssertEqual(result.text, (0..<windows).map { "token\(40 + $0)" }.joined(), file: file, line: line)
        XCTAssertEqual(result.segments.map(\.seek), (0..<windows).map { $0 * 32 }, file: file, line: line)
        XCTAssertEqual(result.language, "en", file: file, line: line)
        XCTAssertEqual(result.timings.totalDecodingWindows, Double(windows), file: file, line: line)
        XCTAssertEqual(fixture.features.calls, windows, file: file, line: line)
        XCTAssertEqual(fixture.encoder.calls, windows, file: file, line: line)
        XCTAssertEqual(fixture.decoder.prepareCalls, 1, file: file, line: line)
        XCTAssertEqual(fixture.decoder.prefillCalls, 1, file: file, line: line)
        XCTAssertEqual(fixture.decoder.inputs.resets.count, windows * attempts, file: file, line: line)
        XCTAssertTrue(fixture.decoder.inputs.resets.allSatisfy { $0 == Reset(prefilled: 1, context: 224) }, file: file, line: line)
        XCTAssertEqual(fixture.decoder.receivedOptions.count, windows * attempts, file: file, line: line)
        let expectedTemperatures = (0..<windows).flatMap { _ in
            (0..<attempts).map { Float($0) * 0.2 }
        }
        XCTAssertEqual(fixture.decoder.temperatures.count, expectedTemperatures.count, file: file, line: line)
        for (actual, expected) in zip(fixture.decoder.temperatures, expectedTemperatures) {
            XCTAssertEqual(actual, expected, accuracy: 0.001, file: file, line: line)
        }
        for options in fixture.decoder.receivedOptions {
            XCTAssertEqual(options.language, "en", file: file, line: line)
            XCTAssertEqual(options.sampleLength, 224, file: file, line: line)
            XCTAssertEqual(options.temperatureFallbackCount, 5, file: file, line: line)
            XCTAssertEqual(options.compressionRatioThreshold, 2.4, file: file, line: line)
            XCTAssertEqual(options.logProbThreshold, -1, file: file, line: line)
            XCTAssertEqual(options.firstTokenLogProbThreshold, -1.5, file: file, line: line)
            XCTAssertTrue(options.usePrefillPrompt, file: file, line: line)
            XCTAssertTrue(options.usePrefillCache, file: file, line: line)
        }
    }
}

private enum ScriptError: Error, Equatable {
    case decoderFailure
    case unexpectedNativePath
    case exhaustedResults
}

private struct Reset: Equatable {
    let prefilled: Int
    let context: Int
}

private final class Inputs: DecodingInputsType {
    var initialPrompt = [101]
    var inputIds: MLMultiArray
    var cacheLength: MLMultiArray
    var resets: [Reset] = []

    init() throws {
        inputIds = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLength = try MLMultiArray(shape: [1], dataType: .int32)
        inputIds[0] = 101
        cacheLength[0] = 0
    }

    func reset(prefilledCacheSize: Int, maxTokenContext: Int) {
        resets.append(Reset(prefilled: prefilledCacheSize, context: maxTokenContext))
        cacheLength[0] = NSNumber(value: prefilledCacheSize)
    }
}

private struct Features: FeatureExtractorOutputType {}
private struct Encoded: AudioEncoderOutputType {}

private final class FeatureExtractorFake: FeatureExtracting {
    var melCount: Int? { 1 }
    var windowSamples: Int? { 32 }
    var calls = 0

    func logMelSpectrogram(fromAudio inputAudio: any AudioProcessorOutputType) async throws -> (any FeatureExtractorOutputType)? {
        calls += 1
        return Features()
    }
}

private final class EncoderFake: AudioEncoding {
    var embedSize: Int? { 1 }
    var calls = 0

    func encodeFeatures(_ features: any FeatureExtractorOutputType) async throws -> (any AudioEncoderOutputType)? {
        calls += 1
        return Encoded()
    }
}

private struct TokenizerFake: WhisperTokenizer {
    let specialTokens = SpecialTokens(
        endToken: 100, englishToken: 102, noSpeechToken: 103, noTimestampsToken: 104,
        specialTokenBegin: 100, startOfPreviousToken: 105, startOfTranscriptToken: 101,
        timeTokenBegin: 200, transcribeToken: 106, translateToken: 107, whitespaceToken: 108
    )
    var allLanguageTokens: Set<Int> { [102] }
    func encode(text: String) -> [Int] { [40] }
    func decode(tokens: [Int]) -> String { tokens.map { "token\($0)" }.joined() }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "token\(id)" }
    func splitToWordTokens(tokenIds: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        (tokenIds.map { "token\($0)" }, tokenIds.map { [$0] })
    }
}

private final class RecordingDecoder: TextDecoding {
    var tokenizer: (any WhisperTokenizer)? = TokenizerFake()
    var prefillData: (any WhisperMLModel)?
    var isModelMultilingual = false
    var supportsWordTimestamps: Bool { false }
    var logitsSize: Int? { 256 }
    var logitsFilters: [any LogitsFiltering]?
    var kvCacheEmbedDim: Int? { 1 }
    var kvCacheMaxSequenceLength: Int? { 224 }
    var windowSize: Int? { 1 }
    var embedSize: Int? { 1 }
    let inputs: Inputs
    let results: [DecodingResult]
    var prepareCalls = 0
    var prefillCalls = 0
    var callbackPresence: [Bool] = []
    var receivedOptions: [DecodingOptions] = []
    var temperatures: [Float] = []
    var decisions: [Bool?] = []
    var retainedCallback: TranscriptionCallback = nil
    var failAtInvocation: Int?
    var cancelAfterFirstResult = false

    init(results: [DecodingResult]) throws {
        self.results = results
        inputs = try Inputs()
    }

    func prepareDecoderInputs(withPrompt initialPrompt: [Int]) throws -> any DecodingInputsType {
        prepareCalls += 1
        inputs.initialPrompt = initialPrompt
        return inputs
    }

    func prefillDecoderInputs(_ decoderInputs: any DecodingInputsType, withOptions options: DecodingOptions?) async throws -> any DecodingInputsType {
        prefillCalls += 1
        inputs.cacheLength[0] = 1
        return inputs
    }

    func predictLogits(_ inputs: any TextDecoderInputType) async throws -> (any TextDecoderOutputType)? {
        throw ScriptError.unexpectedNativePath
    }

    func decodeText(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: any TokenSampling,
        options decoderOptions: DecodingOptions,
        callback: TranscriptionCallback
    ) async throws -> DecodingResult {
        callbackPresence.append(callback != nil)
        receivedOptions.append(decoderOptions)
        XCTAssertTrue((decoderInputs as? Inputs) === inputs)
        let sampler = try XCTUnwrap(tokenSampler as? GreedyTokenSampler)
        temperatures.append(Float(sampler.temperature))
        try Task.checkCancellation()
        if callbackPresence.count == failAtInvocation {
            throw ScriptError.decoderFailure
        }
        retainedCallback = callback
        decisions.append(callback?(Self.progress()))
        let index = callbackPresence.count - 1
        guard results.indices.contains(index) else { throw ScriptError.exhaustedResults }
        if cancelAfterFirstResult, index == 0 {
            withUnsafeCurrentTask { $0?.cancel() }
        }
        return results[index]
    }

    static func progress() -> TranscriptionProgress {
        var timings = TranscriptionTimings()
        timings.totalDecodingLoops = 17
        return TranscriptionProgress(
            timings: timings, text: "synthetic progress", tokens: [7, 8],
            temperature: 0.4, avgLogprob: -0.5, compressionRatio: 1.2, windowId: 777
        )
    }
}

private struct Fixture {
    let features = FeatureExtractorFake()
    let encoder = EncoderFake()
    let decoder: RecordingDecoder
    let audio: [Float]
    let options: DecodingOptions

    init(windows: Int, attempts: Int) throws {
        let options = DecodingOptions(language: "en", detectLanguage: false, skipSpecialTokens: true, windowClipTime: 0)
        self.options = options
        audio = Array(repeating: 0, count: windows * 32)
        var results: [DecodingResult] = []
        for window in 0..<windows {
            for attempt in 0..<attempts {
                var result = DecodingResult.emptyResults
                result.language = "en"
                result.tokens = attempt == attempts - 1 ? [40 + window] : [1]
                result.tokenLogProbs = result.tokens.map { [$0: -0.5] }
                result.avgLogProb = -0.5
                if attempt < attempts - 1 {
                    result.fallback = DecodingFallback(
                        options: options, isFirstTokenLogProbTooLow: attempt % 3 == 0,
                        noSpeechProb: 0, compressionRatio: attempt % 3 == 1 ? 3 : 1,
                        avgLogProb: attempt % 3 == 2 ? -2 : -0.5
                    )
                }
                results.append(result)
            }
        }
        decoder = try RecordingDecoder(results: results)
    }

    func makeTask() -> TranscribeTask {
        TranscribeTask(
            currentTimings: TranscriptionTimings(), progress: nil,
            audioEncoder: encoder, featureExtractor: features, segmentSeeker: SegmentSeeker(),
            textDecoder: decoder, tokenizer: TokenizerFake()
        )
    }

    func run(callback: TranscriptionCallback = nil) async throws -> TranscriptionResult {
        try await makeTask().run(audioArray: audio, decodeOptions: options, callback: callback)
    }
}
