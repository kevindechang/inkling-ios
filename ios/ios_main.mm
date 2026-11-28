// Minimal iOS CLI runner for inkling-ios (launched via `xcrun devicectl device process launch`).
// Commands:
//   info                    print iOS thermal state and available memory
//   probe                   print JSON device state (battery, low_power, thermal, avail_mb)
//   gen <inkling_run args>  run Inkling 2 / varloop inference on device
//   rm <relpath>            delete a file under Documents/
// Demo mode (launch with no args, `gen ... -demo`, or INKLING_DEMO=1): on-screen UI that streams the generated text,
// demo sampling defaults, and the app stays open after generation unless INKLING_DEMO_EXIT=1.
// With `-script <file>` / INKLING_DEMO_SCRIPT (or models/demo_script.txt when tapped) the demo is a multi-turn chat
// that keeps the KV cache between turns.
#import <UIKit/UIKit.h>
#include <os/proc.h>
#include <pthread.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>

int inkling_gen(int argc, char ** argv);

// Demo UI hook implemented in inkling_run.cpp (called on the inference thread).
// kinds: 0 status/progress, 1 prompt, 2 answer text chunk, 3 final stats, 4 thinking progress (text is always empty, n_tok = thinking
//        tokens so far; shown as one "thinking\u2026 N tok" line),
//        5 turn begin (script mode: text=user turn, n_tok=turn index 1-based, progress=turn count),
//        6 turn end (n_tok=decoded tokens, tok_s, progress=TTFT s). In script mode kind 3 carries total tokens, average steady
//        tok/s and peak RAM; its text is "done" or a "stopped: ..." note.
// tok_s is the steady decode rate (all generated tokens, thinking + answer, after the first 3 of the turn / their summed per-token llama_decode time), 0 = not
// available yet (shown as "—"). Time to first token (s, turn prefill start -> first generated token) arrives in `progress` of
// status "generating" (once known), kinds 2/4 (0 = not yet known), kind 6 and the single-turn kind 3.
typedef void (*inkling_ui_cb)(int kind, const char * text, int n_tok, double tok_s, double ram_mb, float progress);
extern "C" void inkling_set_ui_callback(inkling_ui_cb cb);
enum { UI_STATUS = 0, UI_PROMPT = 1, UI_TEXT = 2, UI_FINAL = 3, UI_THINK = 4, UI_TURN_BEGIN = 5, UI_TURN_END = 6 };

static int g_argc;
static char ** g_argv;
static bool g_demo = false;          // on-screen UI + demo defaults
static bool g_demo_noargs = false;   // launched by tapping the icon: synthesized `gen` command
static std::vector<std::string> g_demo_args_store;
static std::vector<char *> g_demo_argv;

static bool env_is_one(const char * name) {
    const char * e = getenv(name);
    return e && strcmp(e, "1") == 0;
}

// Decide demo mode from the command line (before the UI is built). Only `gen` can be a demo.
static void detect_demo() {
    if (g_argc <= 1) {
        g_demo = g_demo_noargs = true;
        g_demo_args_store = {"gen", "-m", "models/inkling2-26b-flashpin-v2-1p-hot40-q2k-i8mm.gguf", "-ngl", "0", "-t", "2", "-tb", "4",
                             "-ub", "64", "-passes", "1", "-k", "4", "-lm", "none", "-lzm", "on", "-c", "8192", "-ctk", "q8_0",
                             "-ctv", "q8_0", "-n", "600"};
        bool have_script = false;   // models/demo_script.txt => multi-turn chat, else the single-turn demo prompt
        @autoreleasepool {
            NSString * docs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
            have_script = docs && [[NSFileManager defaultManager] fileExistsAtPath:[docs stringByAppendingPathComponent:@"models/demo_script.txt"]];
        }
        if (have_script) { g_demo_args_store.push_back("-script"); g_demo_args_store.push_back("models/demo_script.txt"); }
        else { g_demo_args_store.push_back("-f"); g_demo_args_store.push_back("models/demo_prompt.txt"); }
        g_demo_args_store.push_back("-demo");
        for (auto & a : g_demo_args_store) g_demo_argv.push_back((char *) a.c_str());
        g_demo_argv.push_back(nullptr);
        g_argc = (int) g_demo_args_store.size();
        g_argv = g_demo_argv.data();
        return;
    }
    const int a0 = (strcmp(g_argv[1], "--") == 0) ? 2 : 1;
    if (a0 >= g_argc || strcmp(g_argv[a0], "gen") != 0) return;
    bool flag = env_is_one("INKLING_DEMO");
    for (int i = a0 + 1; i < g_argc; i++) if (strcmp(g_argv[i], "-demo") == 0) flag = true;
    g_demo = flag;
}

static void demo_setenv() {
    if (g_demo_noargs) {
        static const char * const kv[][2] = {
            {"INKLING_ALLOW_ONE_PASS", "1"}, {"INKLING_GREEDY_SKIP_SOFTCAP", "1"}, {"GGML_SITU_FAST", "accel"},
            {"GGML_MMID_CACHE_MB", "2950"}, {"GGML_MMID_PIN_N", "40"}, {"GGML_MMID_OVERLAP", "1"}, {"GGML_MMID_REPACK", "1"},
            {"GGML_MMID_LRU", "1"}, {"GGML_MMID_IO_THREADS", "4"}, {"GGML_MMID_STATS", "1"}, {"GGML_MMID_CECAR_PREFILL", "1"},
            {"GGML_MMID_CECAR_BONUS", "0.35"}, {"LLAMA_LM_HEAD_SHORTLIST", "off"},
        };
        for (auto & e : kv) setenv(e[0], e[1], 0);
    }
    static const char * const samp[][2] = {
        {"INKLING_TEMP", "0.35"}, {"INKLING_TOP_K", "40"}, {"INKLING_TOP_P", "0.9"}, {"INKLING_MIN_P", "0.05"},
        {"INKLING_REPEAT_PENALTY", "1.08"}, {"INKLING_REPEAT_LAST_N", "128"},
    };
    for (auto & e : samp) setenv(e[0], e[1], 0);   // explicit env (and -temp etc. args) still win
}

// ---------------------------------------------------------------------------------------------------------
// Demo UI
// ---------------------------------------------------------------------------------------------------------
@interface InklingDemoVC : UIViewController
- (void)handleKind:(int)kind text:(NSString *)text nTok:(int)n tokS:(double)tokS ramMB:(double)ram progress:(float)p;
@end

@implementation InklingDemoVC {
    UILabel * _stats;
    UIScrollView * _scroll;
    UIStackView * _chat;
    UILabel * _modelLabel;                    // answer bubble of the turn in flight
    UILabel * _footer;                        // "NNN tok · decode XX.X tok/s (excl. first 3) · TTFT X.XX s" under that bubble
    NSMutableAttributedString * _modelAttr;
    BOOL _modelHasText;
    BOOL _hasThink;                           // a "thinking\u2026 N tok" line leads the model bubble
    int _thinkTok;
    NSString * _status;
    NSString * _finalNote;
    int _nTok, _turn, _turnCount, _turnsDone;
    double _tokS, _ram, _ttft;
    float _progress;
    bool _done, _script;
}

- (UIStatusBarStyle)preferredStatusBarStyle { return UIStatusBarStyleLightContent; }

- (UILabel *)makeLabel:(UIFont *)font color:(UIColor *)color lines:(NSInteger)lines {
    UILabel * l = [UILabel new];
    l.translatesAutoresizingMaskIntoConstraints = NO;
    l.font = font;
    l.textColor = color;
    l.numberOfLines = lines;
    l.lineBreakMode = NSLineBreakByTruncatingTail;
    return l;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.overrideUserInterfaceStyle = UIUserInterfaceStyleDark;
    UIColor * bg = [UIColor colorWithWhite:0.04 alpha:1.0];
    UIColor * fg = [UIColor colorWithWhite:0.94 alpha:1.0];
    UIColor * dim = [UIColor colorWithWhite:0.58 alpha:1.0];
    self.view.backgroundColor = bg;

    UILabel * title = [self makeLabel:[UIFont systemFontOfSize:24 weight:UIFontWeightBold] color:fg lines:1];
    title.text = @"Inkling 2 FlashPin 26B";
    title.adjustsFontSizeToFitWidth = YES;
    UILabel * sub = [self makeLabel:[UIFont systemFontOfSize:14] color:dim lines:1];
    sub.text = @"iPhone 15 Pro \u00b7 on-device \u00b7 CPU";
    _stats = [self makeLabel:[UIFont monospacedDigitSystemFontOfSize:13 weight:UIFontWeightMedium]
                       color:[UIColor colorWithRed:0.45 green:0.85 blue:0.60 alpha:1.0] lines:3];
    _stats.text = @"loading 0%";

    _scroll = [UIScrollView new];
    _scroll.translatesAutoresizingMaskIntoConstraints = NO;
    _scroll.alwaysBounceVertical = YES;
    _scroll.indicatorStyle = UIScrollViewIndicatorStyleWhite;
    _chat = [UIStackView new];
    _chat.translatesAutoresizingMaskIntoConstraints = NO;
    _chat.axis = UILayoutConstraintAxisVertical;
    _chat.alignment = UIStackViewAlignmentFill;
    _chat.spacing = 12;
    [_scroll addSubview:_chat];

    for (UIView * v in @[title, sub, _stats, _scroll]) [self.view addSubview:v];
    UILayoutGuide * g = self.view.safeAreaLayoutGuide;
    const CGFloat pad = 20;
    [NSLayoutConstraint activateConstraints:@[
        [title.topAnchor constraintEqualToAnchor:g.topAnchor constant:12],
        [title.leadingAnchor constraintEqualToAnchor:g.leadingAnchor constant:pad],
        [title.trailingAnchor constraintEqualToAnchor:g.trailingAnchor constant:-pad],
        [sub.topAnchor constraintEqualToAnchor:title.bottomAnchor constant:2],
        [sub.leadingAnchor constraintEqualToAnchor:title.leadingAnchor],
        [sub.trailingAnchor constraintEqualToAnchor:title.trailingAnchor],
        [_stats.topAnchor constraintEqualToAnchor:sub.bottomAnchor constant:14],
        [_stats.leadingAnchor constraintEqualToAnchor:title.leadingAnchor],
        [_stats.trailingAnchor constraintEqualToAnchor:title.trailingAnchor],
        [_scroll.topAnchor constraintEqualToAnchor:_stats.bottomAnchor constant:10],
        [_scroll.leadingAnchor constraintEqualToAnchor:title.leadingAnchor],
        [_scroll.trailingAnchor constraintEqualToAnchor:title.trailingAnchor],
        [_scroll.bottomAnchor constraintEqualToAnchor:g.bottomAnchor constant:-8],
        [_chat.topAnchor constraintEqualToAnchor:_scroll.contentLayoutGuide.topAnchor constant:4],
        [_chat.bottomAnchor constraintEqualToAnchor:_scroll.contentLayoutGuide.bottomAnchor constant:-12],
        [_chat.leadingAnchor constraintEqualToAnchor:_scroll.contentLayoutGuide.leadingAnchor],
        [_chat.trailingAnchor constraintEqualToAnchor:_scroll.contentLayoutGuide.trailingAnchor],
        [_chat.widthAnchor constraintEqualToAnchor:_scroll.frameLayoutGuide.widthAnchor],
    ]];
    _status = @"loading";
}

static NSString * extract_user_prompt(NSString * raw) {
    NSString * open = @"<|message_user|><|content_text|>";
    NSRange r = [raw rangeOfString:open options:NSBackwardsSearch];
    if (r.location != NSNotFound) {
        NSUInteger from = NSMaxRange(r);
        NSRange e = [raw rangeOfString:@"<|end_message|>" options:0 range:NSMakeRange(from, raw.length - from)];
        NSString * q = [raw substringWithRange:NSMakeRange(from, e.location == NSNotFound ? raw.length - from : e.location - from)];
        q = [q stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
        if (q.length) return q;
    }
    NSString * s = [raw stringByReplacingOccurrencesOfString:@"<\\|[^|>]*\\|>" withString:@" "
                                                     options:NSRegularExpressionSearch range:NSMakeRange(0, raw.length)];
    return [s stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
}

static NSString * fmt_decode(double v) { return v > 0 ? [NSString stringWithFormat:@"%.1f", v] : @"\u2014"; }
static NSString * fmt_ttft(double v) { return v > 0 ? [NSString stringWithFormat:@"%.2f", v] : @"\u2014"; }

- (void)refreshStats {
    NSString * line;
    if ([_status hasPrefix:@"error"]) {
        line = _status;
    } else if (_done && _script) {
        NSString * head = (_finalNote.length && ![_finalNote isEqualToString:@"done"]) ? _finalNote : @"done";
        line = [NSString stringWithFormat:@"%@ \u00b7 %d turns \u00b7 decode avg %@ tok/s (all tokens, excl. first 3/turn) \u00b7 peak RAM %.0f MB",
                head, _turnsDone, fmt_decode(_tokS), _ram];
    } else if (_done) {
        line = [NSString stringWithFormat:@"done \u00b7 %d tok \u00b7 decode %@ tok/s (all tokens, excl. first 3) \u00b7 TTFT %@ s \u00b7 RAM %.0f MB",
                _nTok, fmt_decode(_tokS), fmt_ttft(_ttft), _ram];
    } else if (_script && [_status isEqualToString:@"prefill"]) {
        line = [NSString stringWithFormat:@"turn %d/%d \u00b7 prefill %.0f%% \u00b7 RAM %.0f MB", _turn, _turnCount, _progress * 100.0f, _ram];
    } else if (_script) {
        line = [NSString stringWithFormat:@"turn %d/%d \u00b7 %d tok \u00b7 decode %@ tok/s \u00b7 TTFT %@ s \u00b7 RAM %.0f MB",
                _turn, _turnCount, _nTok, fmt_decode(_tokS), fmt_ttft(_ttft), _ram];
    } else if ([_status isEqualToString:@"loading"] || [_status isEqualToString:@"prefill"]) {
        line = [NSString stringWithFormat:@"%@ %.0f%% \u00b7 RAM %.0f MB", _status, _progress * 100.0f, _ram];
    } else {
        line = [NSString stringWithFormat:@"%@ \u00b7 %d tok \u00b7 decode %@ tok/s \u00b7 TTFT %@ s \u00b7 RAM %.0f MB",
                _status, _nTok, fmt_decode(_tokS), fmt_ttft(_ttft), _ram];
    }
    _stats.text = line;
}

// ---- chat view ----

- (void)scrollToBottom {
    if (_scroll.isDragging || _scroll.isDecelerating) return;   // do not fight the user's finger
    [self.view layoutIfNeeded];
    const UIEdgeInsets in = _scroll.adjustedContentInset;
    const CGFloat y = _scroll.contentSize.height - _scroll.bounds.size.height + in.bottom;
    if (y > -in.top) [_scroll setContentOffset:CGPointMake(0, y) animated:NO];
}

- (UIView *)bubbleWith:(UILabel *)label color:(UIColor *)color {
    UIView * b = [UIView new];
    b.translatesAutoresizingMaskIntoConstraints = NO;
    b.backgroundColor = color;
    b.layer.cornerRadius = 14;
    b.layer.masksToBounds = YES;
    [b addSubview:label];
    [NSLayoutConstraint activateConstraints:@[
        [label.topAnchor constraintEqualToAnchor:b.topAnchor constant:10],
        [label.bottomAnchor constraintEqualToAnchor:b.bottomAnchor constant:-10],
        [label.leadingAnchor constraintEqualToAnchor:b.leadingAnchor constant:14],
        [label.trailingAnchor constraintEqualToAnchor:b.trailingAnchor constant:-14],
    ]];
    return b;
}

- (NSAttributedString *)dimString:(NSString *)s {
    return [[NSAttributedString alloc] initWithString:s attributes:@{
        NSForegroundColorAttributeName: [UIColor colorWithWhite:0.52 alpha:1.0],
        NSFontAttributeName: [UIFont italicSystemFontOfSize:14]}];
}

- (void)addUserBubble:(NSString *)text {
    UILabel * l = [self makeLabel:[UIFont systemFontOfSize:17] color:UIColor.whiteColor lines:0];
    l.lineBreakMode = NSLineBreakByWordWrapping;
    l.text = text;
    UIView * b = [self bubbleWith:l color:[UIColor colorWithRed:0.14 green:0.24 blue:0.42 alpha:1.0]];
    UIView * row = [UIView new];
    [row addSubview:b];
    [NSLayoutConstraint activateConstraints:@[
        [b.topAnchor constraintEqualToAnchor:row.topAnchor],
        [b.bottomAnchor constraintEqualToAnchor:row.bottomAnchor],
        [b.trailingAnchor constraintEqualToAnchor:row.trailingAnchor],
        [b.leadingAnchor constraintGreaterThanOrEqualToAnchor:row.leadingAnchor constant:48],
    ]];
    [_chat addArrangedSubview:row];
}

- (void)addModelBubble {
    _modelLabel = [self makeLabel:[UIFont systemFontOfSize:17] color:UIColor.whiteColor lines:0];
    _modelLabel.lineBreakMode = NSLineBreakByWordWrapping;
    _modelLabel.attributedText = [self dimString:@"\u2026"];
    UIView * b = [self bubbleWith:_modelLabel color:[UIColor colorWithWhite:0.14 alpha:1.0]];
    _footer = [self makeLabel:[UIFont monospacedDigitSystemFontOfSize:12 weight:UIFontWeightRegular]
                        color:[UIColor colorWithWhite:0.5 alpha:1.0] lines:1];
    _footer.numberOfLines = 0;
    _footer.lineBreakMode = NSLineBreakByWordWrapping;
    _footer.hidden = YES;
    UIStackView * col = [[UIStackView alloc] initWithArrangedSubviews:@[b, _footer]];
    col.translatesAutoresizingMaskIntoConstraints = NO;
    col.axis = UILayoutConstraintAxisVertical;
    col.alignment = UIStackViewAlignmentFill;
    col.spacing = 4;
    UIView * row = [UIView new];
    [row addSubview:col];
    [NSLayoutConstraint activateConstraints:@[
        [col.topAnchor constraintEqualToAnchor:row.topAnchor],
        [col.bottomAnchor constraintEqualToAnchor:row.bottomAnchor],
        [col.leadingAnchor constraintEqualToAnchor:row.leadingAnchor],
        [col.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-28],
    ]];
    [_chat addArrangedSubview:row];
    _modelAttr = [NSMutableAttributedString new];
    _modelHasText = NO;
    _hasThink = NO; _thinkTok = 0;
}

- (void)beginTurn:(int)k of:(int)n user:(NSString *)user {
    _turn = k; _turnCount = n; _nTok = 0; _tokS = 0; _ttft = 0;
    [self addUserBubble:user ?: @""];
    [self addModelBubble];
    [self scrollToBottom];
}

- (void)finishTurnTok:(int)n tokS:(double)tokS ttft:(double)ttft {
    if (!_modelLabel) return;
    if (!_modelHasText) _modelLabel.attributedText = [self dimString:@"(empty reply)"];
    _footer.text = [NSString stringWithFormat:@"%d tok \u00b7 decode %@ tok/s (all tokens, excl. first 3) \u00b7 TTFT %@ s",
                    n, fmt_decode(tokS), fmt_ttft(ttft)];
    _footer.hidden = NO;
    _modelLabel = nil; _footer = nil; _modelAttr = nil;
    [self scrollToBottom];
}

- (void)addNote:(NSString *)note {
    UILabel * l = [self makeLabel:[UIFont systemFontOfSize:13] color:[UIColor colorWithWhite:0.5 alpha:1.0] lines:0];
    l.lineBreakMode = NSLineBreakByWordWrapping;
    l.textAlignment = NSTextAlignmentCenter;
    l.text = note;
    [_chat addArrangedSubview:l];
    [self scrollToBottom];
}

// Model bubble = optional dim "thinking\u2026 N tok" line (updated in place, thinking text is never shown) + streamed answer.
- (void)renderModel {
    if (!_modelLabel) return;
    NSMutableAttributedString * out = [NSMutableAttributedString new];
    if (_hasThink) [out appendAttributedString:[self dimString:[NSString stringWithFormat:@"thinking\u2026 %d tok", _thinkTok]]];
    if (_modelAttr.length) {
        if (_hasThink) [out appendAttributedString:[[NSAttributedString alloc] initWithString:@"\n" attributes:@{
            NSFontAttributeName: [UIFont systemFontOfSize:17]}]];
        [out appendAttributedString:_modelAttr];
    }
    _modelLabel.attributedText = out;
    [self scrollToBottom];
}

- (void)updateThinking:(int)n {
    if (!_modelLabel) return;
    _hasThink = YES; _thinkTok = n; _modelHasText = YES;
    [self renderModel];
}

- (void)appendChunk:(NSString *)chunk {
    if (!_modelLabel) return;
    NSDictionary * attrs = @{NSForegroundColorAttributeName: [UIColor colorWithWhite:0.94 alpha:1.0],
                             NSFontAttributeName: [UIFont systemFontOfSize:17]};
    [_modelAttr appendAttributedString:[[NSAttributedString alloc] initWithString:chunk attributes:attrs]];
    _modelHasText = YES;
    [self renderModel];
}

- (void)handleKind:(int)kind text:(NSString *)text nTok:(int)n tokS:(double)tokS ramMB:(double)ram progress:(float)p {
    switch (kind) {
        case UI_PROMPT:   // single-turn demo: one user bubble + one reply
            [self beginTurn:1 of:1 user:extract_user_prompt(text ?: @"")];
            return;
        case UI_TURN_BEGIN:
            _script = true;
            [self beginTurn:n of:(int) (p + 0.5f) user:text];
            _status = @"prefill";
            _progress = 0;
            if (ram > 0) _ram = ram;
            break;
        case UI_TURN_END:
            _turnsDone++;
            _ttft = p;
            [self finishTurnTok:n tokS:tokS ttft:p];
            if (ram > 0) _ram = ram;
            break;
        case UI_STATUS:
            _status = text ?: @"";
            _progress = p;
            if (ram > 0) _ram = ram;
            if (n > 0) _nTok = n;
            if (tokS > 0) _tokS = tokS;
            if ([_status isEqualToString:@"generating"] && p > 0) _ttft = p;   // TTFT, sent once the first token exists
            break;
        case UI_TEXT:
        case UI_THINK:
            if (kind == UI_THINK) [self updateThinking:n];   // n = thinking tokens so far; no text
            else if (text.length) [self appendChunk:text];
            _status = @"generating";
            _nTok = n; _tokS = tokS; if (ram > 0) _ram = ram;
            if (p > 0) _ttft = p;
            break;
        case UI_FINAL:
            _done = true;
            _finalNote = text;
            _nTok = n; _tokS = tokS; if (ram > 0) _ram = ram;
            if (!_script) { if (p > 0) _ttft = p; [self finishTurnTok:n tokS:tokS ttft:_ttft]; }
            else if (text.length && ![text isEqualToString:@"done"]) [self addNote:text];
            break;
    }
    [self refreshStats];
}
@end

static InklingDemoVC * g_demo_vc = nil;

// Runs on the inference thread: copy, then hop to the main queue. Never blocks on the main thread.
static void demo_ui_cb(int kind, const char * text, int n_tok, double tok_s, double ram_mb, float progress) {
    NSString * s = text ? ([NSString stringWithUTF8String:text] ?: @"") : nil;
    InklingDemoVC * vc = g_demo_vc;
    dispatch_async(dispatch_get_main_queue(), ^{
        [vc handleKind:kind text:s nTok:n_tok tokS:tok_s ramMB:ram_mb progress:progress];
    });
}

static const char * thermal_str() {
    switch ([[NSProcessInfo processInfo] thermalState]) {
        case NSProcessInfoThermalStateNominal:  return "nominal";
        case NSProcessInfoThermalStateFair:     return "fair";
        case NSProcessInfoThermalStateSerious:  return "serious";
        case NSProcessInfoThermalStateCritical: return "critical";
    }
    return "unknown";
}

static bool start_big_thread(void * (*fn)(void *), void * arg) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);
    pthread_t th;
    int rc = pthread_create(&th, &attr, fn, arg);
    pthread_attr_destroy(&attr);
    if (rc == 0) pthread_detach(th);
    return rc == 0;
}

static void * run_cli(void *) {
    int rc = 0;
    @autoreleasepool {
        NSString * docs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
        chdir(docs.fileSystemRepresentation);
        printf("system.thermal: %s\n", thermal_str());
        int a0 = g_demo_noargs ? 0 : (g_argc > 1 && strcmp(g_argv[1], "--") == 0) ? 2 : 1;
        if (a0 >= g_argc) {
            printf("usage: info | probe | gen <args...> | rm <path>\n");
            printf("gen sampling/demo args: -temp F -top-k N -top-p F -min-p F -repeat-penalty F -repeat-last-n N -seed N -demo -script <file>\n");
        } else {
            const char * cmd = g_argv[a0];
            if (strcmp(cmd, "info") == 0) {
                printf("avail_mb: %.0f\n", os_proc_available_memory() / 1048576.0);
            } else if (strcmp(cmd, "probe") == 0) {
                dispatch_sync(dispatch_get_main_queue(), ^{
                    UIDevice * d = UIDevice.currentDevice;
                    d.batteryMonitoringEnabled = YES;
                    UIApplicationState st = UIApplication.sharedApplication.applicationState;
                    const char * app_st = st == UIApplicationStateActive ? "active"
                                        : st == UIApplicationStateInactive ? "inactive" : "background";
                    const char * bat = d.batteryState == UIDeviceBatteryStateCharging ? "charging"
                                     : d.batteryState == UIDeviceBatteryStateFull ? "full"
                                     : d.batteryState == UIDeviceBatteryStateUnplugged ? "unplugged" : "unknown";
                    printf("{\"app\":\"%s\",\"battery\":\"%s\",\"battery_level\":%.2f,\"low_power\":%s,\"thermal\":\"%s\",\"avail_mb\":%.0f}\n",
                           app_st, bat, d.batteryLevel,
                           NSProcessInfo.processInfo.lowPowerModeEnabled ? "true" : "false",
                           thermal_str(), os_proc_available_memory() / 1048576.0);
                });
            } else if (strcmp(cmd, "gen") == 0) {
                if (g_demo) {
                    demo_setenv();
                    inkling_set_ui_callback(demo_ui_cb);
                }
                rc = inkling_gen(g_argc - a0, g_argv + a0);
            } else if (strcmp(cmd, "rm") == 0 && a0 + 1 < g_argc) {
                NSString * path = [NSString stringWithUTF8String:g_argv[a0 + 1]];
                NSError * err = nil;
                BOOL ok = [[NSFileManager defaultManager] removeItemAtPath:path error:&err];
                fprintf(stderr, "[rm] %s %s%s\n", g_argv[a0 + 1], ok ? "deleted" : "FAILED: ",
                        ok ? "" : err.localizedDescription.UTF8String);
                rc = ok ? 0 : 1;
            } else {
                fprintf(stderr, "unknown command: %s\n", cmd);
                rc = 1;
            }
        }
        printf("system.thermal.end: %s\n", thermal_str());
    }
    fflush(nullptr);
    if (g_demo) {
        if (rc != 0) {
            char msg[64];
            snprintf(msg, sizeof msg, "error rc=%d", rc);
            demo_ui_cb(UI_STATUS, msg, 0, 0.0, 0.0, 0.0f);
        }
        if (!env_is_one("INKLING_DEMO_EXIT")) return nullptr;   // keep the app (and the screen) open for recording
        usleep(300 * 1000);                                      // let the last UI update land
    }
    _exit(rc);
}

@interface InklingAppDelegate : UIResponder <UIApplicationDelegate>
@property(nonatomic, strong) UIWindow * window;
@end

@implementation InklingAppDelegate
- (BOOL)application:(UIApplication *)app didFinishLaunchingWithOptions:(NSDictionary *)opts {
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    UIViewController * vc;
    if (g_demo) {
        g_demo_vc = [InklingDemoVC new];
        vc = g_demo_vc;
    } else {
        vc = [UIViewController new];
        vc.view.backgroundColor = UIColor.systemBackgroundColor;
    }
    self.window.rootViewController = vc;
    [self.window makeKeyAndVisible];
    app.idleTimerDisabled = YES;
    if (!start_big_thread(run_cli, nullptr)) {
        fprintf(stderr, "cannot start CLI thread\n");
        _exit(1);
    }
    return YES;
}
@end

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    g_argc = argc;
    g_argv = argv;
    detect_demo();
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([InklingAppDelegate class]));
    }
}
