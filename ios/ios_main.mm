// Minimal iOS CLI runner for inkling-ios (launched via `xcrun devicectl device process launch`).
// Commands:
//   info                    print iOS thermal state and available memory
//   probe                   print JSON device state (battery, low_power, thermal, avail_mb)
//   gen <inkling_run args>  run Inkling 2 / varloop inference on device
//   rm <relpath>            delete a file under Documents/
// Demo mode (launch with no args, `gen ... -demo`, or INKLING_DEMO=1): on-screen UI that streams the generated text,
// demo sampling defaults, and the app stays open after generation unless INKLING_DEMO_EXIT=1.
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
// kinds: 0 status/progress, 1 prompt, 2 answer text chunk, 3 final stats, 4 thinking text chunk
typedef void (*inkling_ui_cb)(int kind, const char * text, int n_tok, double tok_s, double ram_mb, float progress);
extern "C" void inkling_set_ui_callback(inkling_ui_cb cb);
enum { UI_STATUS = 0, UI_PROMPT = 1, UI_TEXT = 2, UI_FINAL = 3, UI_THINK = 4 };

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
                             "-ctv", "q8_0", "-n", "400", "-f", "models/demo_prompt.txt", "-demo"};
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
        {"INKLING_TEMP", "0.7"}, {"INKLING_TOP_K", "40"}, {"INKLING_TOP_P", "0.9"}, {"INKLING_MIN_P", "0.05"},
        {"INKLING_REPEAT_PENALTY", "1.1"}, {"INKLING_REPEAT_LAST_N", "128"},
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
    UILabel * _prompt;
    UITextView * _text;
    NSString * _status;
    int _nTok;
    double _tokS, _ram;
    float _progress;
    bool _done;
    int _lastKind;
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
                       color:[UIColor colorWithRed:0.45 green:0.85 blue:0.60 alpha:1.0] lines:2];
    _stats.text = @"loading 0%";
    _prompt = [self makeLabel:[UIFont systemFontOfSize:15] color:dim lines:5];
    _text = [UITextView new];
    _text.translatesAutoresizingMaskIntoConstraints = NO;
    _text.editable = NO;
    _text.backgroundColor = UIColor.clearColor;
    _text.textColor = fg;
    _text.font = [UIFont systemFontOfSize:17];
    _text.textContainerInset = UIEdgeInsetsMake(8, 0, 8, 0);
    _text.textContainer.lineFragmentPadding = 0;
    _text.alwaysBounceVertical = YES;
    _text.indicatorStyle = UIScrollViewIndicatorStyleWhite;

    for (UIView * v in @[title, sub, _stats, _prompt, _text]) [self.view addSubview:v];
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
        [_prompt.topAnchor constraintEqualToAnchor:_stats.bottomAnchor constant:14],
        [_prompt.leadingAnchor constraintEqualToAnchor:title.leadingAnchor],
        [_prompt.trailingAnchor constraintEqualToAnchor:title.trailingAnchor],
        [_text.topAnchor constraintEqualToAnchor:_prompt.bottomAnchor constant:10],
        [_text.leadingAnchor constraintEqualToAnchor:title.leadingAnchor],
        [_text.trailingAnchor constraintEqualToAnchor:title.trailingAnchor],
        [_text.bottomAnchor constraintEqualToAnchor:g.bottomAnchor constant:-8],
    ]];
    _status = @"loading";
    _lastKind = -1;
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

- (void)refreshStats {
    NSString * line;
    if (_done) {
        line = [NSString stringWithFormat:@"done \u00b7 %.1f tok/s \u00b7 RAM %.0f MB", _tokS, _ram];
    } else if ([_status isEqualToString:@"loading"] || [_status isEqualToString:@"prefill"]) {
        line = [NSString stringWithFormat:@"%@ %.0f%% \u00b7 RAM %.0f MB", _status, _progress * 100.0f, _ram];
    } else if ([_status hasPrefix:@"error"]) {
        line = _status;
    } else {
        line = [NSString stringWithFormat:@"%@ \u00b7 %d tok \u00b7 %.1f tok/s \u00b7 RAM %.0f MB", _status, _nTok, _tokS, _ram];
    }
    _stats.text = line;
}

- (void)appendChunk:(NSString *)chunk thinking:(BOOL)thinking {
    UIColor * color = thinking ? [UIColor colorWithWhite:0.52 alpha:1.0] : [UIColor colorWithWhite:0.94 alpha:1.0];
    UIFont * font = [UIFont systemFontOfSize:17];
    NSMutableAttributedString * add = [NSMutableAttributedString new];
    int kind = thinking ? UI_THINK : UI_TEXT;
    if (_lastKind != kind) {
        if (thinking) {
            [add appendAttributedString:[[NSAttributedString alloc] initWithString:@"thinking\u2026 "
                attributes:@{NSForegroundColorAttributeName: color, NSFontAttributeName: font}]];
        } else if (_lastKind == UI_THINK) {
            [add appendAttributedString:[[NSAttributedString alloc] initWithString:@"\n\n"
                attributes:@{NSForegroundColorAttributeName: color, NSFontAttributeName: font}]];
        }
        _lastKind = kind;
    }
    [add appendAttributedString:[[NSAttributedString alloc] initWithString:chunk
        attributes:@{NSForegroundColorAttributeName: color, NSFontAttributeName: font}]];
    [_text.textStorage appendAttributedString:add];
    [_text scrollRangeToVisible:NSMakeRange(_text.textStorage.length, 0)];
}

- (void)handleKind:(int)kind text:(NSString *)text nTok:(int)n tokS:(double)tokS ramMB:(double)ram progress:(float)p {
    switch (kind) {
        case UI_PROMPT:
            _prompt.text = extract_user_prompt(text ?: @"");
            return;
        case UI_STATUS:
            _status = text ?: @"";
            _progress = p;
            if (ram > 0) _ram = ram;
            if (n > 0) _nTok = n;
            if (tokS > 0) _tokS = tokS;
            break;
        case UI_TEXT:
        case UI_THINK:
            if (text.length) [self appendChunk:text thinking:kind == UI_THINK];
            _status = @"generating";
            _nTok = n; _tokS = tokS; if (ram > 0) _ram = ram;
            break;
        case UI_FINAL:
            _done = true;
            _nTok = n; _tokS = tokS; if (ram > 0) _ram = ram;
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
            printf("gen sampling/demo args: -temp F -top-k N -top-p F -min-p F -repeat-penalty F -repeat-last-n N -seed N -demo\n");
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
