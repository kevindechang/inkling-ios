// Minimal iOS CLI runner for inkling-ios (launched via `xcrun devicectl device process launch`).
// Commands:
//   info                    print iOS thermal state and available memory
//   probe                   print JSON device state (battery, low_power, thermal, avail_mb)
//   gen <inkling_run args>  run Inkling 2 / varloop inference on device
//   rm <relpath>            delete a file under Documents/
#import <UIKit/UIKit.h>
#include <os/proc.h>
#include <pthread.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

int inkling_gen(int argc, char ** argv);

static int g_argc;
static char ** g_argv;

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
        int a0 = (g_argc > 1 && strcmp(g_argv[1], "--") == 0) ? 2 : 1;
        if (a0 >= g_argc) {
            printf("usage: info | probe | gen <args...> | rm <path>\n");
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
    _exit(rc);
}

@interface InklingAppDelegate : UIResponder <UIApplicationDelegate>
@property(nonatomic, strong) UIWindow * window;
@end

@implementation InklingAppDelegate
- (BOOL)application:(UIApplication *)app didFinishLaunchingWithOptions:(NSDictionary *)opts {
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    UIViewController * vc = [UIViewController new];
    vc.view.backgroundColor = UIColor.systemBackgroundColor;
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
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([InklingAppDelegate class]));
    }
}
