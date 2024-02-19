#include "common.h"
#include "whisper.h"

#include <unistd.h>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


std::mutex whisper_mutex;
struct whisper_context * g_ctx = nullptr;

typedef struct {
    int64_t start_tm;
    int64_t end_tm;
    std::string text;
}segment_t;

typedef struct
{
    bool success;
    std::vector<segment_t> segments;
}transcribe_result_t;

// Terminal color map. 10 colors grouped in ranges [0.0, 0.1, ..., 0.9]
// Lowest is red, middle is yellow, highest is green.
const std::vector<std::string> k_colors = {
    "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
    "\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
};

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma = false) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

int timestamp_to_sample(int64_t t, int n_samples) {
    return std::max(0, std::min((int) n_samples - 1, (int) ((t*WHISPER_SAMPLE_RATE)/100)));
}

// helper function to replace substrings
void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    for (size_t pos = 0; ; pos += replace.length()) {
        pos = s.find(search, pos);
        if (pos == std::string::npos) break;
        s.erase(pos, search.length());
        s.insert(pos, replace);
    }
}

// command-line parameters
struct whisper_params {
    int32_t n_threads    = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors =  1;
    int32_t offset_t_ms  =  0;
    int32_t offset_n     =  0;
    int32_t duration_ms  =  0;
    int32_t progress_step =  5;
    int32_t max_context  = -1;
    int32_t max_len      =  0;
    int32_t best_of      = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size    = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;

    float word_thold    =  0.01f;
    float entropy_thold =  2.40f;
    float logprob_thold = -1.00f;

    bool speed_up        = false;
    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool output_txt      = false;
    bool output_vtt      = false;
    bool output_srt      = false;
    bool output_wts      = false;
    bool output_csv      = false;
    bool output_jsn      = false;
    bool output_jsn_full = false;
    bool output_lrc      = false;
    bool print_special   = false;
    bool print_colors    = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool log_score       = false;
    bool use_gpu         = true;

    std::string language  = "en";
    std::string prompt;
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model     = "models/ggml-base.en.bin";

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

    std::string openvino_encode_device = "CPU";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};
};


struct whisper_print_user_data {
    const whisper_params * params;

    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

std::string estimate_diarization_speaker(std::vector<std::vector<float>> pcmf32s, int64_t t0, int64_t t1, bool id_only = false) {
    std::string speaker = "";
    const int64_t n_samples = pcmf32s[0].size();

    const int64_t is0 = timestamp_to_sample(t0, n_samples);
    const int64_t is1 = timestamp_to_sample(t1, n_samples);

    double energy0 = 0.0f;
    double energy1 = 0.0f;

    for (int64_t j = is0; j < is1; j++) {
        energy0 += fabs(pcmf32s[0][j]);
        energy1 += fabs(pcmf32s[1][j]);
    }

    if (energy0 > 1.1*energy1) {
        speaker = "0";
    } else if (energy1 > 1.1*energy0) {
        speaker = "1";
    } else {
        speaker = "?";
    }

    //printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, speaker = %s\n", is0, is1, energy0, energy1, speaker.c_str());

    if (!id_only) {
        speaker.insert(0, "(speaker ");
        speaker.append(")");
    }

    return speaker;
}

void whisper_print_progress_callback(struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) {
    int progress_step = ((whisper_print_user_data *) user_data)->params->progress_step;
    int * progress_prev  = &(((whisper_print_user_data *) user_data)->progress_prev);
    if (progress >= *progress_prev + progress_step) {
        *progress_prev += progress_step;
        fprintf(stderr, "%s: progress = %3d%%\n", __func__, progress);
    }
}

void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
    const auto & params  = *((whisper_print_user_data *) user_data)->params;
    const auto & pcmf32s = *((whisper_print_user_data *) user_data)->pcmf32s;

    const int n_segments = whisper_full_n_segments(ctx);

    std::string speaker = "";

    int64_t t0 = 0;
    int64_t t1 = 0;

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0) {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++) {
        if (!params.no_timestamps || params.diarize) {
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps) {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.diarize && pcmf32s.size() == 2) {
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
        }

        if (params.print_colors) {
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                if (params.print_special == false) {
                    const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                    if (id >= whisper_token_eot(ctx)) {
                        continue;
                    }
                }

                const char * text = whisper_full_get_token_text(ctx, i, j);
                const float  p    = whisper_full_get_token_p   (ctx, i, j);

                const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) (std::pow(p, 3)*float(k_colors.size()))));

                printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
            }
        } else {
            const char * text = whisper_full_get_segment_text(ctx, i);

            printf("%s%s", speaker.c_str(), text);
        }

        if (params.tinydiarize) {
            if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                printf("%s", params.tdrz_speaker_turn.c_str());
            }
        }

        // with timestamps or speakers: each segment on new line
        if (!params.no_timestamps || params.diarize) {
            printf("\n");
        }

        fflush(stdout);
    }
}

// Load model
int load(const char *model_path) {
    struct whisper_context_params cparams;
    cparams.use_gpu = true;

    whisper_mutex.lock();
    g_ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (g_ctx == nullptr) {
        whisper_mutex.unlock();
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return -1;
    }

    whisper_mutex.unlock();
    printf("load model %s successfully\n", model_path);
    return 0;
}

// Transcribe audio to text
transcribe_result_t transcribe(const char* audio_path, const char* language = "auto", int beam_size = 5) {
    whisper_params params;
    std::vector<float> pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

    params.language = language;
    params.beam_size = beam_size;

    transcribe_result_t result;
    result.success = false;

    whisper_mutex.lock();
    // read audio
    if (!::read_wav(audio_path, pcmf32, pcmf32s, params.diarize)) {
        fprintf(stderr, "error: failed to read WAV file '%s'\n", audio_path);
        whisper_mutex.unlock();
        return result;
    }

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(g_ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        if (params.detect_language) {
            params.language = "auto";
        }
        fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, %d beams + best of %d, lang = %s, task = %s, %stimestamps = %d ...\n",
                __func__, audio_path, int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE,
                params.n_threads, params.n_processors, params.beam_size, params.best_of,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.tinydiarize ? "tdrz = 1, " : "",
                params.no_timestamps ? 0 : 1);
    }

    // inference
    {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

        wparams.print_realtime   = false;
        wparams.print_progress   = params.print_progress;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.print_special    = params.print_special;
        wparams.translate        = params.translate;
        wparams.language         = params.language.c_str();
        wparams.detect_language  = params.detect_language;
        wparams.n_threads        = params.n_threads;
        wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
        wparams.offset_ms        = params.offset_t_ms;
        wparams.duration_ms      = params.duration_ms;

        wparams.token_timestamps = params.output_wts || params.output_jsn_full || params.max_len > 0;
        wparams.thold_pt         = params.word_thold;
        wparams.max_len          = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
        wparams.split_on_word    = params.split_on_word;

        wparams.speed_up         = params.speed_up;
        wparams.debug_mode       = params.debug_mode;

        wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

        wparams.initial_prompt   = params.prompt.c_str();

        wparams.greedy.best_of        = params.best_of;
        wparams.beam_search.beam_size = params.beam_size;

        wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
        wparams.entropy_thold    = params.entropy_thold;
        wparams.logprob_thold    = params.logprob_thold;

        whisper_print_user_data user_data = { &params, &pcmf32s, 0 };

        // this callback is called on each new segment
        if (!wparams.print_realtime) {
            wparams.new_segment_callback           = whisper_print_segment_callback;
            wparams.new_segment_callback_user_data = &user_data;
        }

        if (wparams.print_progress) {
            wparams.progress_callback           = whisper_print_progress_callback;
            wparams.progress_callback_user_data = &user_data;
        }

        // examples for abort mechanism
        // in examples below, we do not abort the processing, but we could if the flag is set to true

        // the callback is called before every encoder run - if it returns false, the processing is aborted
        {
            static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

            wparams.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
                bool is_aborted = *(bool*)user_data;
                return !is_aborted;
            };
            wparams.encoder_begin_callback_user_data = &is_aborted;
        }

        // the callback is called before every computation - if it returns true, the computation is aborted
        {
            static bool is_aborted = false; // NOTE: this should be atomic to avoid data race

            wparams.abort_callback = [](void * user_data) {
                bool is_aborted = *(bool*)user_data;
                return is_aborted;
            };
            wparams.abort_callback_user_data = &is_aborted;
        }


        if (whisper_full_parallel(g_ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
            whisper_mutex.unlock();
            fprintf(stderr, "%s: failed to process audio\n", audio_path);
            result.success = false;
            return result;
        }
    }

    // parse result
    const int n_segments = whisper_full_n_segments(g_ctx);
    int64_t t0 = 0;
    int64_t t1 = 0;

    for (int i = 0; i < n_segments; i++)
    {
        t0 = whisper_full_get_segment_t0(g_ctx, i);
        t1 = whisper_full_get_segment_t1(g_ctx, i);
        const char* text = whisper_full_get_segment_text(g_ctx, i);
        segment_t segment = {
            .start_tm   = t0 * 10,  // ms
            .end_tm     = t1 * 10,  // ms
            .text       = text,
        };
        result.segments.emplace_back(segment);
    }

    whisper_mutex.unlock();
    result.success = true;
    return result;
}

void destroy(void) {
    if (g_ctx != nullptr) {
        whisper_mutex.lock();
        whisper_free(g_ctx);
        g_ctx = nullptr;
        whisper_mutex.unlock();
        printf("destroy whisper context!\n");
    }
}

// int main(int argc, char **argv) {
//     const char *model_path = "/data/models/whisper/ggml/base.bin";
//     const char *audio_path = "/home/rd/Downloads/en-cn.wav";
//     transcribe_result_t result;

//     load(model_path);
//     result = transcribe(audio_path);
//     printf("transcribe success: %d\n", result.success);
//     for (int i = 0; i < result.segments.size(); i++) {
//         printf("start_tm: %ld, end_tm: %ld, text: %s\n", 
//             result.segments[i].start_tm, result.segments[i].end_tm, result.segments[i].text.c_str());
//     }
//     destroy();
//     return 0;
// }

bool is_initialized(void) {
    return g_ctx != nullptr;
}

namespace py = pybind11;
PYBIND11_MODULE(pywhisper, m) {
    m.doc() = "wrapping whisper.cpp as dynamic library, expose load, transcribe and destroy functions.";

    // 导出transcribe_result_t和segment_t两个结构体
    // 否则，调用transribe函数会报错。
    py::class_<transcribe_result_t>(m, "transcribe_result_t")
        .def(py::init<>())
        .def_readwrite("success", &transcribe_result_t::success)
        .def_readwrite("segments", &transcribe_result_t::segments);

    // include pybind11/stl.h头文件，可以自动将std::vector转换成Python类型
    // 否则，访问segments时会报错。
    // 参考：https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
    py::class_<segment_t>(m, "segment_t")
        .def(py::init<>())
        .def_readwrite("start_tm", &segment_t::start_tm)
        .def_readwrite("end_tm", &segment_t::end_tm)
        .def_readwrite("text", &segment_t::text);

    m.def("is_initialized", &is_initialized, "whether whisiper context is initialized");
    m.def("load", &load, py::arg("model_path"), "load whisper model");
    m.def("transcribe", &transcribe,
        py::arg("audio_path"), py::arg("langugage") = "auto", py::arg("beam_size") = 5,
        "transcribe audio to text");
    m.def("destroy", &destroy, "destroy whisper context");
}
