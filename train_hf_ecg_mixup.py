"""
Hugging Face ECG Image Classification — MixUp / CutMix Augmentation Variant
-----------------------------------------------------------------------------

This script is an enhanced version of train_hf_ecg.py that adds MixUp and CutMix
data augmentation to combat overfitting on the small ECG training set (~500 images).

** What's new vs. train_hf_ecg.py **
  - MixUp: blends two images and their labels with a Beta-sampled coefficient λ
  - CutMix: replaces a random rectangular patch in one image with a patch from another
  - AUG_MODE controls which strategy is used: "mixup", "cutmix", or "both"
  - MixUpCutMixCollator applies augmentation at batch assembly time (training only)
  - MixUpCutMixTrainer uses soft-label cross-entropy loss (required for mixed labels)
  - Evaluation is NEVER augmented — standard hard-label accuracy is preserved

** Pushed to **
  Branch : mixup-cutmix-aug
  HF repo: MMM0003/ecg-classifier-mixup

** How to Run on Kaggle (FREE T4 GPU) **
1. Go to https://www.kaggle.com/code → click "New Notebook"
2. On the right sidebar → Session Options → Accelerator → "GPU T4 x2"
3. Go to Settings (top right) → Add Secret:
       Key:   HF_TOKEN
       Value: (your Hugging Face token with WRITE access)
       ✅ Check "Attach to this notebook"
4. In Cell 1, run:
       !pip install -q transformers datasets evaluate accelerate scikit-learn torchvision
5. In Cell 2:
       !git clone -b mixup-cutmix-aug https://github.com/MahimaMaryMathew/HRV_Analysis_Platform.git
6. In Cell 3:
       !python HRV_Analysis_Platform/train_hf_ecg_mixup.py

** How to Run on Google Colab **
1. Change Runtime → T4 GPU
2. In Cell 1: !pip install -q transformers datasets evaluate accelerate scikit-learn torchvision
3. In Cell 2: from huggingface_hub import notebook_login; notebook_login()
4. In Cell 3: !git clone -b mixup-cutmix-aug https://github.com/MahimaMaryMathew/HRV_Analysis_Platform.git
5. In Cell 4: !python HRV_Analysis_Platform/train_hf_ecg_mixup.py
"""

import os
import glob
import shutil
import time
import random
import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from datasets import load_dataset, Dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

# Increase network timeouts BEFORE any HF dataset calls.
import os, time  # noqa: E401
os.environ.setdefault("HF_DATASETS_HTTP_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "120")
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "warning")

# ==============================================================================
# Configuration
# ==============================================================================
DATASET_ID   = "edcci/GenECG"
MODEL_ID     = "google/vit-base-patch16-224-in21k"

HF_USERNAME  = "MMM0003"
# Separate model name to preserve the baseline ecg-classifier-10
HUB_MODEL_ID = f"{HF_USERNAME}/ecg-classifier-mixup"

OUTPUT_DIR   = "./ecg-mixup-results"
MAX_RETRIES  = 3

# Balanced sampling cap (per class)
MAX_SAMPLES_PER_CLASS = 100

# Training epochs — slightly higher than baseline; augmentation buys us more
# useful gradient signal per epoch, so 15 epochs generalises better than 10.
NUM_EPOCHS   = 15

# --------------------------------------------------------------------------
# Augmentation settings
# --------------------------------------------------------------------------
# AUG_MODE: "mixup"  → only MixUp
#            "cutmix" → only CutMix
#            "both"   → randomly choose MixUp or CutMix for each batch
AUG_MODE      = "both"

# Beta distribution alpha parameter — higher = more aggressive mixing.
# 0.4 is a good default; literature uses 0.2–1.0.
MIXUP_ALPHA   = 0.4
CUTMIX_ALPHA  = 0.4
# --------------------------------------------------------------------------

USE_GOOGLE_DRIVE    = False
GDRIVE_BACKUP_DIR   = "/content/drive/MyDrive/ecg-mixup-checkpoints"


# ==============================================================================
# HF Token setup
# ==============================================================================
def setup_hf_token():
    """Automatically find and set the HF_TOKEN."""
    try:
        from kaggle_secrets import UserSecretsClient
        secret = UserSecretsClient().get_secret("HF_TOKEN")
        if secret:
            os.environ["HF_TOKEN"] = secret
            print("🔑 HF_TOKEN loaded from Kaggle Secrets")
            return True
    except Exception:
        pass

    if os.environ.get("HF_TOKEN"):
        print("🔑 HF_TOKEN found in environment")
        return True

    try:
        from huggingface_hub import login
        print("⚠️  HF_TOKEN not found. Attempting interactive login...")
        login()
        return True
    except Exception:
        pass

    print("❌ Could not find HF_TOKEN. Please set it as a Kaggle Secret or environment variable.")
    return False


# ==============================================================================
# Helper utilities (unchanged from baseline)
# ==============================================================================

def get_device():
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("⚠️  No GPU detected — using CPU. Training will be slower.")
    return "cpu"


def get_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]


def setup_google_drive():
    if not USE_GOOGLE_DRIVE:
        return
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        os.makedirs(GDRIVE_BACKUP_DIR, exist_ok=True)
        print(f"📁 Google Drive backup enabled → {GDRIVE_BACKUP_DIR}")
    except ImportError:
        print("⚠️  google.colab not available — Google Drive backup disabled.")
    except Exception as e:
        print(f"⚠️  Could not mount Google Drive: {e}")


def backup_checkpoints_to_drive():
    if not USE_GOOGLE_DRIVE:
        return
    try:
        if os.path.exists(OUTPUT_DIR):
            for ckpt in glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")):
                dest = os.path.join(GDRIVE_BACKUP_DIR, os.path.basename(ckpt))
                if not os.path.exists(dest):
                    shutil.copytree(ckpt, dest)
                    print(f"   💾 Backed up {os.path.basename(ckpt)} → Google Drive")
    except Exception as e:
        print(f"⚠️  Drive backup failed (non-fatal): {e}")


def restore_checkpoints_from_drive():
    if not USE_GOOGLE_DRIVE:
        return
    try:
        if get_latest_checkpoint(OUTPUT_DIR) is not None:
            return
        drive_ckpts = glob.glob(os.path.join(GDRIVE_BACKUP_DIR, "checkpoint-*"))
        if not drive_ckpts:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for ckpt in drive_ckpts:
            dest = os.path.join(OUTPUT_DIR, os.path.basename(ckpt))
            if not os.path.exists(dest):
                shutil.copytree(ckpt, dest)
                print(f"   📥 Restored {os.path.basename(ckpt)} from Google Drive")
    except Exception as e:
        print(f"⚠️  Drive restore failed (non-fatal): {e}")


def move_model_to_device(model, device):
    try:
        model.to(device)
        print(f"   Model moved to {device}")
    except Exception as e:
        print(f"⚠️  Could not move model to {device}: {e}")
        model.to("cpu")
        print("   Fell back to CPU")


# ==============================================================================
# MixUp & CutMix augmentation functions
# ==============================================================================

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer label tensor to one-hot float tensor."""
    return F.one_hot(labels, num_classes=num_classes).float()


def mixup_batch(
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MixUp augmentation to a batch.

    For each image i, we pick a random partner j and compute:
      λ  ~ Beta(alpha, alpha)
      x' = λ · x_i + (1-λ) · x_j
      y' = λ · y_i + (1-λ) · y_j   (soft one-hot)

    Args:
        pixel_values : (B, C, H, W) float tensor
        labels       : (B,) integer tensor
        alpha        : Beta distribution concentration parameter
        num_classes  : number of ECG diagnostic classes

    Returns:
        mixed_pixels : (B, C, H, W) float tensor
        mixed_labels : (B, num_classes) soft label tensor
    """
    B = pixel_values.size(0)
    lam = float(np.random.beta(alpha, alpha))

    # Random permutation as mixing partner index
    idx = torch.randperm(B, device=pixel_values.device)

    mixed_pixels = lam * pixel_values + (1.0 - lam) * pixel_values[idx]

    y_a = one_hot(labels, num_classes)
    y_b = one_hot(labels[idx], num_classes)
    mixed_labels = lam * y_a + (1.0 - lam) * y_b

    return mixed_pixels, mixed_labels


def cutmix_batch(
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply CutMix augmentation to a batch.

    For each image i, we pick a random partner j, sample a rectangular mask
    covering fraction (1-λ) of the image, and paste that region from j → i:
      x' = x_i with patch from x_j
      λ_adj = 1 - (patch_area / total_area)
      y'    = λ_adj · y_i + (1-λ_adj) · y_j

    Args:
        pixel_values : (B, C, H, W) float tensor
        labels       : (B,) integer tensor
        alpha        : Beta distribution concentration parameter
        num_classes  : number of ECG diagnostic classes

    Returns:
        mixed_pixels : (B, C, H, W) float tensor
        mixed_labels : (B, num_classes) soft label tensor
    """
    B, C, H, W = pixel_values.shape
    lam = float(np.random.beta(alpha, alpha))

    idx = torch.randperm(B, device=pixel_values.device)

    # Sample a random bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = random.randint(0, W)
    cy = random.randint(0, H)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    mixed_pixels = pixel_values.clone()
    mixed_pixels[:, :, y1:y2, x1:x2] = pixel_values[idx, :, y1:y2, x1:x2]

    # Adjust λ based on actual patch area
    lam_adj = 1.0 - float((x2 - x1) * (y2 - y1)) / float(H * W)

    y_a = one_hot(labels, num_classes)
    y_b = one_hot(labels[idx], num_classes)
    mixed_labels = lam_adj * y_a + (1.0 - lam_adj) * y_b

    return mixed_pixels, mixed_labels


# ==============================================================================
# Custom Collator: applies augmentation during training batches
# ==============================================================================

class MixUpCutMixCollator:
    """
    Drop-in replacement for DefaultDataCollator that optionally applies
    MixUp or CutMix during training.

    Set `training=True` when used as the train collator.
    Set `training=False` (or use DefaultDataCollator) for evaluation.
    """

    def __init__(
        self,
        aug_mode: str,
        mixup_alpha: float,
        cutmix_alpha: float,
        num_classes: int,
        training: bool = True,
    ):
        self.aug_mode   = aug_mode
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes  = num_classes
        self.training   = training

    def __call__(self, features: list[dict]) -> dict:
        # Stack pixel_values and labels from the feature list
        pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in features])
        labels       = torch.tensor([f["label"] for f in features], dtype=torch.long)

        if not self.training or len(features) < 2:
            # Evaluation: no augmentation; return hard labels
            return {"pixel_values": pixel_values, "labels": labels}

        # Choose augmentation strategy
        if self.aug_mode == "mixup":
            strategy = "mixup"
        elif self.aug_mode == "cutmix":
            strategy = "cutmix"
        else:
            # "both": randomly pick one strategy per batch
            strategy = random.choice(["mixup", "cutmix"])

        if strategy == "mixup":
            pixel_values, soft_labels = mixup_batch(
                pixel_values, labels, self.mixup_alpha, self.num_classes
            )
        else:
            pixel_values, soft_labels = cutmix_batch(
                pixel_values, labels, self.cutmix_alpha, self.num_classes
            )

        return {"pixel_values": pixel_values, "labels": soft_labels}


# ==============================================================================
# Custom Trainer: soft-label cross-entropy loss
# ==============================================================================

class MixUpCutMixTrainer(Trainer):
    """
    Trainer subclass that:
      1. Uses soft-label cross-entropy during training (MixUp/CutMix labels are
         continuous distributions, not one-hot integers).
      2. Overrides get_eval_dataloader so evaluation always uses DefaultDataCollator
         (hard integer labels), keeping eval_accuracy metrics reliable.

    Loss logic:
      labels.dim() == 2  →  soft-label CE: -sum(y_soft * log_softmax(logits))
      labels.dim() == 1  →  standard CE: cross_entropy(logits, labels)
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits  # (B, num_classes)

        if labels.dim() == 2:
            # Soft-label cross-entropy (training with MixUp/CutMix)
            log_probs = F.log_softmax(logits, dim=-1)     # (B, num_classes)
            loss = -(labels * log_probs).sum(dim=-1).mean()
        else:
            # Standard hard-label cross-entropy (eval fallback)
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Force the evaluation dataloader to use DefaultDataCollator so that
        labels remain hard integers (required for accuracy metric computation).
        """
        original_collator = self.data_collator
        self.data_collator = DefaultDataCollator()
        dataloader = super().get_eval_dataloader(eval_dataset)
        self.data_collator = original_collator
        return dataloader


# ==============================================================================
# Main Training Function
# ==============================================================================

def run_training():
    # ------------------------------------------------------------------
    # 0. Authenticate with Hugging Face
    # ------------------------------------------------------------------
    if not setup_hf_token():
        print("Cannot proceed without HF_TOKEN. Exiting.")
        return

    # ------------------------------------------------------------------
    # 1. Device Detection & Drive Setup
    # ------------------------------------------------------------------
    device = get_device()
    setup_google_drive()
    restore_checkpoints_from_drive()

    print(f"\n🎛  Augmentation : {AUG_MODE.upper()}  "
          f"(MixUp α={MIXUP_ALPHA}, CutMix α={CUTMIX_ALPHA})")

    # ------------------------------------------------------------------
    # 2. Load Dataset with CORRECT diagnostic labels
    # ------------------------------------------------------------------
    print(f"\n📦 Loading dataset and PTB-XL metadata...")
    try:
        import pandas as pd
        import ast

        # Step 1: Download PTB-XL metadata for diagnostic labels
        print("   Downloading PTB-XL metadata (scp_codes → superclass mapping)...")
        PTBXL_CSV_URL = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
        SCP_CSV_URL   = "https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv"

        ptbxl_df = pd.read_csv(PTBXL_CSV_URL)
        scp_df   = pd.read_csv(SCP_CSV_URL, index_col=0)

        scp_diag = scp_df[scp_df['diagnostic'] == 1.0]
        code_to_superclass = {}
        for code, row in scp_diag.iterrows():
            sc = row.get('diagnostic_class', '')
            if pd.notna(sc) and sc:
                code_to_superclass[code] = sc

        SUPERCLASS_NAMES  = ["NORM", "MI", "STTC", "CD", "HYP"]
        superclass_to_id  = {s: i for i, s in enumerate(SUPERCLASS_NAMES)}

        def get_superclass_id(scp_codes_str):
            try:
                codes    = ast.literal_eval(scp_codes_str)
                max_code = max(codes, key=codes.get)
                sc       = code_to_superclass.get(max_code, None)
                if sc and sc in superclass_to_id:
                    return superclass_to_id[sc]
            except Exception:
                pass
            return -1

        ptbxl_df['superclass_id'] = ptbxl_df['scp_codes'].apply(get_superclass_id)
        ecg_id_to_superclass = {
            int(row['ecg_id']) - 1: row['superclass_id']
            for _, row in ptbxl_df.iterrows()
        }

        valid_count = sum(1 for v in ecg_id_to_superclass.values() if v >= 0)
        print(f"   ✅ Mapped {valid_count}/{len(ecg_id_to_superclass)} records to superclasses")
        for sc in SUPERCLASS_NAMES:
            count = sum(1 for v in ecg_id_to_superclass.values()
                        if v == superclass_to_id[sc])
            print(f"      {sc}: {count}")

        # Step 2: Balanced streaming
        print(f"\n   Scanning full GenECG stream (cap: {MAX_SAMPLES_PER_CLASS}/class)...")

        stream = load_dataset(DATASET_ID, split="train", streaming=True)

        per_class     = {i: [] for i in range(len(SUPERCLASS_NAMES))}
        skipped       = 0
        scanned       = 0
        MAX_STREAM_RETRIES = 10
        stream_attempts    = 0
        done               = False

        while not done and stream_attempts < MAX_STREAM_RETRIES:
            try:
                stream = load_dataset(DATASET_ID, split="train", streaming=True)
                if scanned > 0:
                    print(f"   ♻️  Resuming stream from record {scanned} "
                          f"(attempt {stream_attempts + 1}/{MAX_STREAM_RETRIES})...")
                    stream = stream.skip(scanned)

                stream_attempts += 1

                for i_rel, sample in enumerate(stream):
                    i = scanned + i_rel
                    scanned += 1

                    superclass_id = ecg_id_to_superclass.get(i, -1)
                    if superclass_id < 0:
                        skipped += 1
                        continue

                    if len(per_class[superclass_id]) < MAX_SAMPLES_PER_CLASS:
                        try:
                            _ = sample["image"]
                            sample['label'] = superclass_id
                            per_class[superclass_id].append(sample)
                        except Exception as img_err:
                            skipped += 1
                            if skipped % 50 == 1:
                                print(f"   ⚠️  Skipped image #{i} "
                                      f"({type(img_err).__name__}: "
                                      f"{str(img_err)[:80]}). "
                                      f"Total skipped: {skipped}")
                            continue

                    if scanned % 500 == 0:
                        collected    = sum(len(v) for v in per_class.values())
                        class_status = " | ".join(
                            f"{SUPERCLASS_NAMES[sc_id]}: {len(bucket)}"
                            for sc_id, bucket in sorted(per_class.items())
                        )
                        print(f"   Scanned {scanned} | Collected {collected}"
                              f" | {class_status}")

                    if all(len(b) >= MAX_SAMPLES_PER_CLASS for b in per_class.values()):
                        print(f"   ✅ All classes capped at {MAX_SAMPLES_PER_CLASS}. Stopping early.")
                        done = True
                        break
                else:
                    done = True

            except Exception as stream_err:
                print(f"\n   ⚠️  Stream error at record {scanned}: "
                      f"{type(stream_err).__name__}: {str(stream_err)[:100]}")
                if stream_attempts < MAX_STREAM_RETRIES:
                    print(f"   ⏳ Reconnecting in 5s and resuming from record {scanned}...")
                    time.sleep(5)
                else:
                    print(f"   ❌ Exceeded {MAX_STREAM_RETRIES} reconnect attempts.")
                    done = True

        print(f"\n   Raw per-class counts after full scan:")
        for sc_id, sc_name in enumerate(SUPERCLASS_NAMES):
            print(f"      {sc_name}: {len(per_class[sc_id])}")

        random.seed(42)
        min_count = min(len(bucket) for bucket in per_class.values())
        print(f"\n   Balancing: downsampling all classes to {min_count} samples each")
        for sc_id in per_class:
            if len(per_class[sc_id]) > min_count:
                per_class[sc_id] = random.sample(per_class[sc_id], min_count)

        samples = [s for bucket in per_class.values() for s in bucket]
        total   = len(samples)
        print(f"   ✅ Final dataset: {total} images ({min_count}/class × {len(SUPERCLASS_NAMES)} classes)")
        print(f"   (Scanned {scanned} records, skipped {skipped} with unknown labels)")

        dataset = Dataset.from_list(samples).shuffle(seed=42)
        dataset = dataset.train_test_split(test_size=0.2)
        print(f"   Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")

    except Exception as e:
        import traceback
        print(f"❌ Failed to load dataset. Error: {e}")
        traceback.print_exc()
        return

    labels     = SUPERCLASS_NAMES
    label2id   = {label: str(i) for i, label in enumerate(labels)}
    id2label   = {str(i): label for i, label in enumerate(labels)}
    NUM_CLASSES = len(labels)
    print(f"   Labels: {id2label}")

    # ------------------------------------------------------------------
    # 3. Preprocessing
    # ------------------------------------------------------------------
    print(f"\n🔧 Loading image processor: {MODEL_ID}")
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)

    def transforms(example_batch):
        images = [x.convert("RGB") for x in example_batch["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["label"] = example_batch["label"]
        return inputs

    prepared_ds = dataset.with_transform(transforms)

    # ------------------------------------------------------------------
    # 4. Load Model
    # ------------------------------------------------------------------
    print(f"\n🧠 Loading model: {MODEL_ID}")
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    move_model_to_device(model, device)

    # ------------------------------------------------------------------
    # 5. Metrics (eval uses hard labels — no augmentation)
    # ------------------------------------------------------------------
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels_arr = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels_arr)

    # ------------------------------------------------------------------
    # 6. Collators
    #    - train_collator : applies MixUp/CutMix augmentation per batch
    #    - eval uses DefaultDataCollator via MixUpCutMixTrainer override
    # ------------------------------------------------------------------
    train_collator = MixUpCutMixCollator(
        aug_mode    = AUG_MODE,
        mixup_alpha = MIXUP_ALPHA,
        cutmix_alpha= CUTMIX_ALPHA,
        num_classes = NUM_CLASSES,
        training    = True,
    )

    # ------------------------------------------------------------------
    # 7. Training Loop with Retry
    # ------------------------------------------------------------------
    attempt          = 0
    training_complete = False

    while attempt < MAX_RETRIES and not training_complete:
        attempt += 1
        current_device = get_device()
        use_fp16       = current_device == "cuda"

        print(f"\n{'='*60}")
        print(f"🚀 Training attempt {attempt}/{MAX_RETRIES}  [{AUG_MODE.upper()} augmentation]")
        print(f"   Device  : {current_device}")
        print(f"   FP16    : {use_fp16}")
        print(f"   Epochs  : {NUM_EPOCHS}")
        print(f"{'='*60}\n")

        move_model_to_device(model, current_device)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=16 if current_device == "cuda" else 8,
            eval_strategy="steps",
            num_train_epochs=NUM_EPOCHS,
            fp16=use_fp16,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=True,
            hub_model_id=HUB_MODEL_ID,
            save_strategy="steps",
        )

        trainer = MixUpCutMixTrainer(
            model=model,
            args=training_args,
            data_collator=train_collator,   # MixUp/CutMix during training
            train_dataset=prepared_ds["train"],
            eval_dataset=prepared_ds["test"],
            processing_class=processor,
            compute_metrics=compute_metrics,
        )
        # Note: MixUpCutMixTrainer.get_eval_dataloader() automatically swaps
        # in DefaultDataCollator for evaluation — no extra patching needed.

        latest_ckpt = get_latest_checkpoint(OUTPUT_DIR)
        if latest_ckpt:
            print(f"♻️  Resuming from checkpoint: {latest_ckpt}")
        else:
            print("🆕 Starting fresh — no checkpoint found.")

        try:
            trainer.train(resume_from_checkpoint=latest_ckpt)
            training_complete = True
            backup_checkpoints_to_drive()

        except RuntimeError as e:
            error_msg  = str(e).lower()
            is_cuda_err = any(
                kw in error_msg
                for kw in ["cuda", "gpu", "device-side assert", "out of memory",
                           "nccl", "cublas", "cudnn"]
            )
            if is_cuda_err and attempt < MAX_RETRIES:
                print(f"\n⚠️  CUDA error during training: {e}")
                print(f"   Falling back to CPU and resuming from last checkpoint...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                backup_checkpoints_to_drive()
                move_model_to_device(model, "cpu")
            else:
                print(f"\n❌ Unrecoverable error during training: {e}")
                raise

        except KeyboardInterrupt:
            print("\n⏸️  Training interrupted by user.")
            print("   Progress is saved. Re-run this script to resume.")
            backup_checkpoints_to_drive()
            return

    if training_complete:
        print("\n✅ Training complete! Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()
        print(f"🎉 Model pushed to: https://huggingface.co/{HUB_MODEL_ID}")
        print(f"   Branch: mixup-cutmix-aug")
        print("   You can now use this model ID in your HRV Analysis Platform!")
    else:
        print(f"\n❌ Training failed after {MAX_RETRIES} attempts.")
        print("   Your latest checkpoint is preserved. Fix the issue and re-run.")


if __name__ == "__main__":
    run_training()
