// VoxCPM2 CLI demo — text-to-speech synthesis
//
// Usage:
//   ./voxcpm2-cli [options] <BaseLM.gguf> <Acoustic.gguf>
//
// Examples:
//   # Basic TTS
//   ./voxcpm2-cli -t "Hello, welcome to VoxCPM2." -o output.wav base.gguf acoustic.gguf
//
//   # Voice design
//   ./voxcpm2-cli -t "(A young woman, gentle voice)Hello there!" -o voice.wav base.gguf acoustic.gguf
//
//   # Voice cloning
//   ./voxcpm2-cli -t "This is a cloned voice." -r speaker.wav -o clone.wav base.gguf acoustic.gguf
//
//   # Streaming
//   ./voxcpm2-cli -t "Streaming test." --stream base.gguf acoustic.gguf

#include "voxcpm2_runtime.h"
#include "common.h"
#include "ggml-backend.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#ifdef _MSC_VER
#pragma comment(lib, "shell32.lib")
#endif
#endif

namespace {

// All strings in this CLI are UTF-8. On Windows argv arrives in the ANSI code
// page, so main() rebuilds it from the wide command line, and files are opened
// through std::filesystem paths built from UTF-8.
static std::filesystem::path utf8_path(const std::string & s) {
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
#else
    return std::filesystem::u8path(s);
#endif
}

// Read a whole UTF-8 file, dropping a leading byte-order mark.
static bool read_text_file(const std::string & path, std::string & out) {
    std::ifstream f(utf8_path(path), std::ios::binary);
    if (!f) {
        LOG_ERR("Cannot open: %s\n", path.c_str());
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    if (out.size() >= 3 && (unsigned char) out[0] == 0xEF && (unsigned char) out[1] == 0xBB && (unsigned char) out[2] == 0xBF) {
        out.erase(0, 3);
    }
    return true;
}

struct CliConfig {
    std::string base_lm_path;
    std::string acoustic_path;
    std::string text;
    std::string text_file;
    std::string batch_file;
    int    n_threads     = 0;  // 0 = physical cores
    std::string reference_wav_path;
    std::string prompt_wav_path;
    std::string prompt_text;
    std::string output_path = "output.wav";
    bool   use_gpu       = true;
    bool   streaming     = false;
    int    max_steps     = 200;
    int    timesteps     = 10;
    float  cfg_value     = 2.0f;
    float  temperature   = 1.0f;
    int    n_gpu_layers  = -1;
    int    seed          = 42;
};

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "VoxCPM2 CLI — Text-to-Speech Synthesis\n"
        "\n"
        "Usage: %s [options] <BaseLM.gguf> <Acoustic.gguf>\n"
        "\n"
        "Required:\n"
        "  <BaseLM.gguf>        Path to VoxCPM2-BaseLM GGUF file\n"
        "  <Acoustic.gguf>      Path to VoxCPM2-Acoustic GGUF file\n"
        "\n"
        "Options:\n"
        "  -t, --text TEXT      Input text to synthesize\n"
        "  --text-file PATH     Read the input text from a UTF-8 file\n"
        "  --batch PATH         Synthesize many clips with one model load. Each line of\n"
        "                       the UTF-8 file is <output.wav> TAB <cfg> TAB <text>;\n"
        "                       an empty cfg uses --cfg. One JSON result line per clip\n"
        "                       is printed to stdout. Exit 2 if any clip failed.\n"
        "  --threads N          CPU threads (default: physical cores)\n"
        "  --version            Print the version and compute devices, then exit\n"
        "  -o, --output PATH    Output WAV file path (default: output.wav)\n"
        "  -r, --reference PATH Reference WAV for voice cloning\n"
        "  --prompt-wav PATH    Prompt WAV for ultimate cloning\n"
        "  --prompt-text TEXT   Transcript of prompt WAV for ultimate cloning\n"
        "  --stream             Streaming mode (output chunks as they arrive)\n"
        "  --steps N            Max decode steps (default: 200)\n"
        "  --timesteps N        CFM inference timesteps (default: 10)\n"
        "  --cfg F              CFG guidance scale (default: 2.0)\n"
        "  --temperature F      Noise temperature (default: 1.0)\n"
        "  --seed N             Random seed (default: 42)\n"
        "  --cpu                Use CPU backend (default: GPU)\n"
        "  --n-gpu-layers N     Number of layers to offload to GPU (default: all)\n"
        "\n"
        "Examples:\n"
        "  # Basic TTS\n"
        "  %s -t \"Hello, welcome to VoxCPM2.\" base.gguf acoustic.gguf\n"
        "\n"
        "  # Voice design (describe voice in parentheses at start of text)\n"
        "  %s -t \"(A young woman, gentle voice)Hello there!\" base.gguf acoustic.gguf\n"
        "\n"
        "  # Voice cloning\n"
        "  %s -t \"Cloned voice text.\" -r speaker.wav base.gguf acoustic.gguf\n"
        "\n",
        prog, prog, prog, prog);
}

#ifndef VOXCPM2_CLI_VERSION
#define VOXCPM2_CLI_VERSION "dev"
#endif

// Print the version and the compute devices, exit 0. A GPU build (compiled
// with VOXCPM2_EXPECT_GPU) exits 3 when it finds no usable GPU, so installers
// can start it once to decide between the GPU and the CPU build.
static int print_version() {
    std::printf("voxcpm2-cli %s\n", VOXCPM2_CLI_VERSION);
    ggml_backend_load_all();
    bool gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const bool is_gpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
        gpu = gpu || is_gpu;
        std::printf("device: %s (%s)%s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), is_gpu ? " gpu" : "");
    }
#ifdef VOXCPM2_EXPECT_GPU
    if (!gpu) {
        std::fprintf(stderr, "no usable GPU: this build needs an NVIDIA GPU and a CUDA 12 driver; use the CPU build\n");
        return 3;
    }
#endif
    return 0;
}

static bool parse_args(int argc, char ** argv, CliConfig & cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "-t" || arg == "--text") {
            if (++i < argc) cfg.text = argv[i];
        } else if (arg == "--text-file") {
            if (++i < argc) cfg.text_file = argv[i];
        } else if (arg == "--batch") {
            if (++i < argc) cfg.batch_file = argv[i];
        } else if (arg == "--threads") {
            if (++i < argc) cfg.n_threads = std::atoi(argv[i]);
        } else if (arg == "-o" || arg == "--output") {
            if (++i < argc) cfg.output_path = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i < argc) cfg.reference_wav_path = argv[i];
        } else if (arg == "--prompt-wav") {
            if (++i < argc) cfg.prompt_wav_path = argv[i];
        } else if (arg == "--prompt-text") {
            if (++i < argc) cfg.prompt_text = argv[i];
        } else if (arg == "--stream") {
            cfg.streaming = true;
        } else if (arg == "--cpu") {
            cfg.use_gpu = false;
        } else if (arg == "--steps") {
            if (++i < argc) cfg.max_steps = std::atoi(argv[i]);
        } else if (arg == "--timesteps") {
            if (++i < argc) cfg.timesteps = std::atoi(argv[i]);
        } else if (arg == "--cfg") {
            if (++i < argc) cfg.cfg_value = static_cast<float>(std::atof(argv[i]));
        } else if (arg == "--temperature") {
            if (++i < argc) cfg.temperature = static_cast<float>(std::atof(argv[i]));
        } else if (arg == "--seed") {
            if (++i < argc) cfg.seed = std::atoi(argv[i]);
        } else if (arg == "--n-gpu-layers") {
            if (++i < argc) cfg.n_gpu_layers = std::atoi(argv[i]);
        } else if (cfg.base_lm_path.empty()) {
            cfg.base_lm_path = arg;
        } else if (cfg.acoustic_path.empty()) {
            cfg.acoustic_path = arg;
        }
    }

    if (cfg.base_lm_path.empty() || cfg.acoustic_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    if (!cfg.text_file.empty()) {
        if (!read_text_file(cfg.text_file, cfg.text)) return false;
        while (!cfg.text.empty() && (cfg.text.back() == '\n' || cfg.text.back() == '\r')) cfg.text.pop_back();
    }
    if (cfg.text.empty() && cfg.batch_file.empty()) {
        LOG_ERR("No input text specified. Use -t \"your text here\", --text-file or --batch\n");
        return false;
    }
    if (cfg.n_threads <= 0) {
        cfg.n_threads = std::max(1, (int) common_cpu_get_num_physical_cores());
    }
    return true;
}

static void write_wav(const std::string & path, const std::vector<float> & pcm, int sample_rate) {
    std::ofstream f(utf8_path(path), std::ios::binary);
    if (!f) {
        LOG_ERR("Cannot create: %s\n", path.c_str());
        return;
    }
    const int32_t n_samples = static_cast<int32_t>(pcm.size());
    const int32_t byte_rate = sample_rate * 2;  // 16-bit mono
    const int32_t data_size = n_samples * 2;
    const int32_t chunk_size = 36 + data_size;

    auto w32 = [&](int32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); };
    auto w16 = [&](int16_t v) { f.write(reinterpret_cast<const char *>(&v), 2); };

    f.write("RIFF", 4); w32(chunk_size); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(1); w32(sample_rate); w32(byte_rate); w16(2); w16(16);
    f.write("data", 4); w32(data_size);
    for (float v : pcm) {
        v = std::max(-1.0f, std::min(1.0f, v));
        w16(static_cast<int16_t>(v * 32767.0f));
    }
}

static std::vector<float> load_wav_mono(const std::string & path, int & out_sample_rate) {
    std::ifstream f(utf8_path(path), std::ios::binary);
    if (!f) {
        LOG_ERR("Cannot open: %s\n", path.c_str());
        return {};
    }
    char riff[4]; int32_t chunk_size; char wave[4];
    f.read(riff, 4); f.read(reinterpret_cast<char *>(&chunk_size), 4); f.read(wave, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
        LOG_ERR("Not a valid WAV file: %s\n", path.c_str());
        return {};
    }
    int16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    int32_t sample_rate = 0, data_size = 0;
    while (f) {
        char id[4]; int32_t size;
        f.read(id, 4); f.read(reinterpret_cast<char *>(&size), 4);
        if (!f) break;
        if (std::strncmp(id, "fmt ", 4) == 0) {
            f.read(reinterpret_cast<char *>(&audio_format), 2);
            f.read(reinterpret_cast<char *>(&num_channels), 2);
            f.read(reinterpret_cast<char *>(&sample_rate), 4);
            int32_t byte_rate, block_align;
            f.read(reinterpret_cast<char *>(&byte_rate), 4);
            f.read(reinterpret_cast<char *>(&block_align), 2);
            f.read(reinterpret_cast<char *>(&bits_per_sample), 2);
            if (size > 16) f.seekg(size - 16, std::ios::cur);
        } else if (std::strncmp(id, "data", 4) == 0) {
            data_size = size;
            break;
        } else {
            f.seekg(size, std::ios::cur);
        }
    }
    if (audio_format != 1 || data_size <= 0) {
        LOG_ERR("Unsupported WAV format: %s (fmt=%d, data=%d)\n", path.c_str(), audio_format, data_size);
        return {};
    }
    out_sample_rate = sample_rate;
    int n_samples = data_size / (bits_per_sample / 8);
    std::vector<int16_t> raw(n_samples);
    f.read(reinterpret_cast<char *>(raw.data()), data_size);

    std::vector<float> pcm(static_cast<size_t>(n_samples) / num_channels);
    for (int i = 0; i < n_samples / num_channels; ++i) {
        float sum = 0.0f;
        for (int ch = 0; ch < num_channels; ++ch) {
            sum += static_cast<float>(raw[i * num_channels + ch]) / 32768.0f;
        }
        pcm[i] = sum / static_cast<float>(num_channels);
    }
    return pcm;
}

static void report_stats(const std::vector<float> & wav, int sample_rate, double elapsed_s) {
    double dur = static_cast<double>(wav.size()) / static_cast<double>(sample_rate);
    double rms = 0.0, peak = 0.0;
    for (float v : wav) {
        rms += static_cast<double>(v * v);
        peak = std::max(peak, static_cast<double>(std::fabs(v)));
    }
    rms = std::sqrt(rms / static_cast<double>(wav.size()));
    LOG_INF("Audio: %.2fs, %zu samples, %d Hz, RMS=%.3f, peak=%.3f\n",
            dur, wav.size(), sample_rate, rms, peak);
    LOG_INF("Elapsed: %.3fs, RTF=%.3f\n", elapsed_s, elapsed_s / dur);
}

struct Clip {
    std::string output_path;
    std::string text;
    float       cfg_value;
};

// Parse a batch file: one clip per line, <output.wav> TAB <cfg> TAB <text>.
// Blank lines and lines starting with '#' are skipped.
static bool read_batch(const CliConfig & cfg, std::vector<Clip> & clips) {
    std::string body;
    if (!read_text_file(cfg.batch_file, body)) return false;
    std::istringstream lines(body);
    std::string line;
    int lineno = 0;
    while (std::getline(lines, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const size_t t1 = line.find('\t');
        const size_t t2 = t1 == std::string::npos ? std::string::npos : line.find('\t', t1 + 1);
        if (t2 == std::string::npos) {
            LOG_ERR("%s:%d: expected <output.wav> TAB <cfg> TAB <text>\n", cfg.batch_file.c_str(), lineno);
            return false;
        }
        Clip c;
        c.output_path = line.substr(0, t1);
        const std::string cfg_col = line.substr(t1 + 1, t2 - t1 - 1);
        c.cfg_value = cfg_col.empty() ? cfg.cfg_value : static_cast<float>(std::atof(cfg_col.c_str()));
        c.text = line.substr(t2 + 1);
        if (c.output_path.empty() || c.text.empty()) {
            LOG_ERR("%s:%d: empty output path or text\n", cfg.batch_file.c_str(), lineno);
            return false;
        }
        clips.push_back(std::move(c));
    }
    if (clips.empty()) {
        LOG_ERR("%s: no clips\n", cfg.batch_file.c_str());
        return false;
    }
    return true;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    for (char ch : s) {
        switch (ch) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char) ch < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char) ch);
                    out += buf;
                } else {
                    out += ch;
                }
        }
    }
    return out;
}

// The voice prompt shared by every clip of a run, loaded once.
struct VoicePrompt {
    enum class Mode { none, clone, continuation } mode = Mode::none;
    std::vector<float> wav;
    int                sample_rate = 0;
};

static bool load_voice_prompt(const CliConfig & cfg, VoicePrompt & vp) {
    if (!cfg.prompt_wav_path.empty() && !cfg.prompt_text.empty()) {
        LOG_INF("Loading prompt WAV: %s\n", cfg.prompt_wav_path.c_str());
        LOG_INF("  Prompt text: \"%s\"\n", cfg.prompt_text.c_str());
        vp.mode = VoicePrompt::Mode::continuation;
        vp.wav  = load_wav_mono(cfg.prompt_wav_path, vp.sample_rate);
    } else if (!cfg.reference_wav_path.empty()) {
        LOG_INF("Loading reference WAV: %s\n", cfg.reference_wav_path.c_str());
        vp.mode = VoicePrompt::Mode::clone;
        vp.wav  = load_wav_mono(cfg.reference_wav_path, vp.sample_rate);
    } else {
        return true;
    }
    if (vp.wav.empty()) {
        LOG_ERR("Failed to load the reference/prompt WAV\n");
        return false;
    }
    LOG_INF("  Reference: %.2fs, %d Hz\n", static_cast<double>(vp.wav.size()) / vp.sample_rate, vp.sample_rate);
    return true;
}

static bool synthesize(VoxCPM2Runtime & runtime, const CliConfig & cfg, const VoicePrompt & vp, const std::string & text,
                       VoxCPM2GenerateParams params, std::vector<float> & wav) {
    auto collect = [&wav](const std::vector<float> & chunk, bool is_final) {
        wav.insert(wav.end(), chunk.begin(), chunk.end());
        LOG_INF("  Chunk: %zu samples%s\n", chunk.size(), is_final ? " (final)" : "");
        return true;
    };
    wav.clear();
    if (vp.mode == VoicePrompt::Mode::continuation) {
        params.reference_sample_rate = vp.sample_rate;
        if (cfg.streaming) {
            if (!runtime.generate_with_continuation_streaming(text, cfg.prompt_text, vp.wav, collect, params)) wav.clear();
        } else {
            wav = runtime.generate_with_continuation(text, cfg.prompt_text, vp.wav, params);
        }
    } else if (vp.mode == VoicePrompt::Mode::clone) {
        params.reference_sample_rate = vp.sample_rate;
        if (cfg.streaming) {
            if (!runtime.generate_with_clone_streaming(text, vp.wav, collect, params)) wav.clear();
        } else {
            wav = runtime.generate_with_clone(text, vp.wav, params);
        }
    } else if (cfg.streaming) {
        runtime.generate_streaming(text, collect, params);
    } else {
        wav = runtime.generate(text, params);
    }
    if (wav.empty()) {
        LOG_ERR("Generation failed: %s\n", runtime.last_error().c_str());
        return false;
    }
    return true;
}

static int run_synthesis(const CliConfig & cfg) {
    std::vector<Clip> clips;
    if (!cfg.batch_file.empty()) {
        if (!read_batch(cfg, clips)) return 1;
    } else {
        clips.push_back({ cfg.output_path, cfg.text, cfg.cfg_value });
    }

    LOG_INF("VoxCPM2 CLI — Text-to-Speech Synthesis\n");
    LOG_INF("  BaseLM:   %s\n", cfg.base_lm_path.c_str());
    LOG_INF("  Acoustic: %s\n", cfg.acoustic_path.c_str());
    LOG_INF("  Clips:    %zu\n", clips.size());
    LOG_INF("  Backend:  %s\n", cfg.use_gpu ? "GPU" : "CPU");
    LOG_INF("  Threads:  %d\n", cfg.n_threads);
    LOG_INF("  Config:   steps=%d timesteps=%d cfg=%.1f temp=%.1f seed=%d\n",
            cfg.max_steps, cfg.timesteps, cfg.cfg_value, cfg.temperature, cfg.seed);

    VoicePrompt vp;
    if (!load_voice_prompt(cfg, vp)) return 1;

    auto t_load = std::chrono::steady_clock::now();
    VoxCPM2Runtime runtime;
    if (!runtime.init(cfg.base_lm_path, cfg.acoustic_path, cfg.n_gpu_layers, cfg.use_gpu)) {
        LOG_ERR("Failed to initialize VoxCPM2 runtime: %s\n", runtime.last_error().c_str());
        return 1;
    }
    runtime.set_n_threads(cfg.n_threads);
    LOG_INF("Model load: %.3fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count());

    const int sample_rate = runtime.sample_rate();
    int failed = 0;
    for (size_t i = 0; i < clips.size(); ++i) {
        const Clip & clip = clips[i];
        LOG_INF("[%zu/%zu] Text: \"%s\"\n", i + 1, clips.size(), clip.text.c_str());

        VoxCPM2GenerateParams params;
        params.max_steps           = cfg.max_steps;
        params.inference_timesteps = cfg.timesteps;
        params.cfg_value           = clip.cfg_value;
        params.temperature         = cfg.temperature;
        params.seed                = static_cast<uint32_t>(cfg.seed);

        std::vector<float> wav;
        auto t_start = std::chrono::steady_clock::now();
        const bool ok = synthesize(runtime, cfg, vp, clip.text, params, wav);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        double seconds = 0.0;
        if (ok) {
            report_stats(wav, sample_rate, elapsed);
            write_wav(clip.output_path, wav, sample_rate);
            LOG_INF("WAV saved: %s\n", clip.output_path.c_str());
            seconds = static_cast<double>(wav.size()) / sample_rate;
        } else {
            ++failed;
        }
        if (!cfg.batch_file.empty()) {
            std::fprintf(stdout, "{\"index\":%zu,\"output\":\"%s\",\"ok\":%s,\"seconds\":%.3f,\"elapsed\":%.3f,\"sampleRate\":%d}\n",
                         i + 1, json_escape(clip.output_path).c_str(), ok ? "true" : "false", seconds, elapsed, sample_rate);
            std::fflush(stdout);
        }
    }

    if (cfg.batch_file.empty()) return failed ? 1 : 0;
    return failed ? 2 : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    ggml_time_init();

#ifdef _WIN32
    // Rebuild argv as UTF-8 from the wide command line.
    std::vector<std::string> utf8_args;
    std::vector<char *>      utf8_argv;
    {
        int      wargc = 0;
        LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
        if (wargv) {
            for (int i = 0; i < wargc; ++i) {
                const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
                std::string s(n > 0 ? n - 1 : 0, '\0');
                if (n > 1) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
                utf8_args.push_back(std::move(s));
            }
            LocalFree(wargv);
            for (auto & s : utf8_args) utf8_argv.push_back(s.data());
            utf8_argv.push_back(nullptr);
            argc = wargc;
            argv = utf8_argv.data();
        }
    }
#endif

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            const int rc = print_version();
            std::fflush(stdout);
            std::fflush(stderr);
            std::_Exit(rc);
        }
    }

    CliConfig cfg;
    if (!parse_args(argc, argv, cfg)) return 1;

    const int rc = run_synthesis(cfg);

    common_log_flush(common_log_main());
    std::fflush(stdout);
    std::fflush(stderr);

    std::_Exit(rc);
}
