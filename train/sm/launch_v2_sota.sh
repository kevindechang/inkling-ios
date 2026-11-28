#!/bin/bash
# launch_v2_sota.sh <tag> <instance-type> <max-runtime-seconds> <hp-json> [s3-input-prefix]
# Submits the WSD-40 v2 QAT job. Local verification is `bash -n` only; do not run this to test.
# SPOT=1 -> managed spot, MaxWaitTimeInSeconds = MAXS + 3600. SPOT=0 -> on-demand.
set -euo pipefail
TAG=$1; ITYPE=$2; MAXS=$3; HP_JSON=$4
: "${SM_BUCKET:?set SM_BUCKET (S3 bucket for code/data/checkpoints)}"
: "${SM_ROLE_ARN:?set SM_ROLE_ARN (SageMaker execution role ARN)}"
S3_IN=${5:-s3://$SM_BUCKET/inkling/data/stage3/}
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN=$(dirname "$HERE")
ROOT=$(dirname "$TRAIN")
export AWS_PROFILE="${AWS_PROFILE:-default}"
S3_PROFILE="${S3_PROFILE:-$AWS_PROFILE}"
SM_PROFILE="${SM_PROFILE:-$AWS_PROFILE}"
BUCKET=$SM_BUCKET
REGION=${AWS_REGION:-us-east-1}
ROLE=$SM_ROLE_ARN
IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training@sha256:c73e4a578c264bda58cf46dd34c97a8b3e1316153fe389032c211a03315803e1
NAME="inkling-v2-$TAG"
SPOT="${SPOT:-0}"

TMP=$(mktemp -d)
mkdir -p "$TMP/src/llama.cpp"
cp "$TRAIN/distill_v2_sota.py" "$HERE/sm_v2_entry.py" "$TRAIN/hot_plan_40.json" "$TMP/src/"
cp -R "$ROOT/third_party/llama.cpp/gguf-py" "$TMP/src/llama.cpp/gguf-py"
tar -C "$TMP/src" -czf "$TMP/source.tar.gz" .
aws s3 cp "$TMP/source.tar.gz" "s3://$BUCKET/inkling/code/v2-$TAG/source.tar.gz" \
  --region "$REGION" --only-show-errors --profile "$S3_PROFILE"
HP=$(python3 -c "
import json, sys
hp = json.loads(sys.argv[1])
hp.update({
    'sagemaker_program': 'sm_v2_entry.py',
    'sagemaker_submit_directory': 's3://$BUCKET/inkling/code/v2-$TAG/source.tar.gz',
})
print(json.dumps({k: str(v) for k, v in hp.items()}))" "$HP_JSON")
STOP="MaxRuntimeInSeconds=$MAXS"
CMD=(aws sagemaker create-training-job --region "$REGION" --profile "$SM_PROFILE" --training-job-name "$NAME")
if [[ "$SPOT" == "1" ]]; then
  CMD+=(--enable-managed-spot-training)
  STOP="MaxRuntimeInSeconds=$MAXS,MaxWaitTimeInSeconds=$((MAXS + 3600))"
fi
CMD+=(
  --algorithm-specification "TrainingImage=$IMAGE,TrainingInputMode=File" --role-arn "$ROLE"
  --input-data-config "[{\"ChannelName\":\"training\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"$S3_IN\",\"S3DataDistributionType\":\"FullyReplicated\"}}}]"
  --resource-config "InstanceType=$ITYPE,InstanceCount=1,VolumeSizeInGB=150"
  --stopping-condition "$STOP"
  --output-data-config "S3OutputPath=s3://$BUCKET/inkling/out/"
  --checkpoint-config "S3Uri=s3://$BUCKET/inkling/ckpt/v2-$TAG/,LocalPath=/opt/ml/checkpoints"
  --hyper-parameters "$HP" --tags Key=project,Value=inkling-ios --query TrainingJobArn --output text
)
"${CMD[@]}"
rm -rf "$TMP"
