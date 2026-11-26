#!/bin/bash
# Build, sign, and install InklingBench.app for iOS (ARM64, CPU-only streaming MoE).
# Required environment variables (unless LIBS_ONLY=1):
#   LLAMA_DIR              path to patched llama.cpp checkout (@ 8a97fa98ab1a454709bbf5ac979661feaeadec40)
#   IOS_UDID               target iPhone CoreDevice UDID (from `xcrun devicectl list devices`)
#   IOS_BUNDLE_ID          bundle identifier matching your provisioning profile
#   IOS_TEAM_ID            Apple Developer Team ID
#   IOS_SIGN_IDENTITY      codesign identity SHA-1 or name
#   IOS_PROVISION_PROFILE  path to .mobileprovision (must carry com.apple.developer.kernel.increased-memory-limit)
# Optional:
#   IOS_I8MM=1             build ARMv8.6 + i8mm (A15+ / A17 Pro; default 1)
#   MIN_IOS=18.0           deployment target
#   REBUILD_LIBS=1         force rebuild of llama.cpp static libs
#   LIBS_ONLY=1            build static libs only (no signing/install)
#   NO_INSTALL=1           build and sign without installing
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=${LLAMA_DIR:-$HERE/../third_party/llama.cpp}
IOS_I8MM=${IOS_I8MM:-1}
MIN_IOS=${MIN_IOS:-18.0}

if [ "$IOS_I8MM" = 1 ]; then
  ARM_ARCH=${ARM_ARCH:-armv8.6-a+dotprod+fp16+i8mm}
  B=${B:-$L/build-ios-i8mm}
else
  ARM_ARCH=${ARM_ARCH:-armv8.4-a+dotprod+fp16}
  B=${B:-$L/build-ios-a13}
fi

if [ ! -f "$B/build.ninja" ]; then
  cmake -S "$L" -B "$B" -G Ninja \
    -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET="$MIN_IOS" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_METAL=OFF -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="$ARM_ARCH" \
    -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF
fi

if [ ! -f "$B/src/libllama.a" ] || [ "${REBUILD_LIBS:-0}" = 1 ]; then
  cmake --build "$B" -j 6 --target ggml-base ggml-cpu ggml llama
fi
[ "${LIBS_ONLY:-0}" = 1 ] && exit 0

: "${IOS_BUNDLE_ID:?set IOS_BUNDLE_ID}"
: "${IOS_TEAM_ID:?set IOS_TEAM_ID}"
: "${IOS_SIGN_IDENTITY:?set IOS_SIGN_IDENTITY}"
: "${IOS_PROVISION_PROFILE:?set IOS_PROVISION_PROFILE}"

APP=$HERE/build/InklingBench.app
mkdir -p "$APP"
SDK=$(xcrun --sdk iphoneos --show-sdk-path)
xcrun --sdk iphoneos clang++ -target arm64-apple-ios"$MIN_IOS" -isysroot "$SDK" -std=c++17 -O2 -fobjc-arc \
  -I"$L/include" -I"$L/ggml/include" "$HERE/ios_main.mm" "$HERE/inkling_run.cpp" \
  "$B/src/libllama.a" "$B/ggml/src/libggml.a" "$B/ggml/src/ggml-cpu/libggml-cpu.a" "$B/ggml/src/libggml-base.a" \
  -framework UIKit -framework Foundation -framework Accelerate \
  -o "$APP/InklingBench"

cat > "$APP/Info.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>${IOS_BUNDLE_ID}</string>
<key>CFBundleExecutable</key><string>InklingBench</string>
<key>CFBundleName</key><string>InklingBench</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>MinimumOSVersion</key><string>${MIN_IOS}</string>
<key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
<key>UIDeviceFamily</key><array><integer>1</integer></array>
<key>UIFileSharingEnabled</key><true/>
<key>UILaunchScreen</key><dict/>
<key>UIRequiredDeviceCapabilities</key><array><string>arm64</string></array>
</dict></plist>
PL

cp "$IOS_PROVISION_PROFILE" "$APP/embedded.mobileprovision"
cat > "$HERE/build/ent.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>application-identifier</key><string>${IOS_TEAM_ID}.${IOS_BUNDLE_ID}</string>
<key>com.apple.developer.team-identifier</key><string>${IOS_TEAM_ID}</string>
<key>com.apple.developer.kernel.increased-memory-limit</key><true/>
<key>get-task-allow</key><true/>
</dict></plist>
PL

codesign --force --sign "$IOS_SIGN_IDENTITY" --entitlements "$HERE/build/ent.plist" "$APP"
codesign -v "$APP" && echo "SIGNED"
if [ "${NO_INSTALL:-0}" != 1 ]; then
  : "${IOS_UDID:?set IOS_UDID (or NO_INSTALL=1)}"
  xcrun devicectl device install app --device "$IOS_UDID" "$APP"
fi
