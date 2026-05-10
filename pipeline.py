"""
pipeline.py
-----------
End-to-end Milestone 2 pipeline.
Runs all stages in sequence from a single entry point.

Stages
------
    1. preprocess   — resize + normalize all splits → data/processed/
    2. augment      — generate augmented train set  → data/processed/train_augmented.npz
    3. features     — extract features for all splits → features/
    4. mrmr         — MRMR feature selection → features/*_mrmr.npz
    5. knn          — train and evaluate KNN
    6. softmax      — train and evaluate Softmax
    7. cnn          — train and evaluate CNN from scratch
    8. convmixer    — train and evaluate ConvMixer

Usage
-----
    # Full pipeline
    python pipeline.py

    # Run specific stages only
    python pipeline.py --stages preprocess augment features mrmr

    # Skip slow stages (if already done)
    python pipeline.py --stages knn softmax cnn convmixer
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'augmentation'))
sys.path.insert(0, os.path.join(ROOT, 'features'))
sys.path.insert(0, os.path.join(ROOT, 'models'))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))
sys.path.insert(0, os.path.join(ROOT, 'optimizer'))

PROCESSED_DIR = os.path.join(ROOT, 'data', 'processed')
FEATURES_DIR  = os.path.join(ROOT, 'features')
LOGS_DIR      = os.path.join(ROOT, 'logs')
CLASS_NAMES   = sorted(['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'])
N_CLASSES     = 6


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def stage_preprocess(size: int = 150):
    section(f"Stage 1: Preprocessing (size={size})")
    from preprocessing import process_split, CLASS_NAMES as CN, CLASS_TO_IDX

    import preprocessing as pp
    pp.TARGET_H = pp.TARGET_W = size
    pp.PROCESSED_DIR = PROCESSED_DIR

    for split in ['train', 'val', 'test']:
        process_split(split, size, size)


def stage_augment(multiplier: int = 3):
    section(f"Stage 2: Augmentation (multiplier={multiplier})")
    from augmentor import load_and_augment_train
    load_and_augment_train(
        processed_dir=PROCESSED_DIR,
        multiplier=multiplier,
        seed=42,
        save=True,
    )


def stage_features():
    section("Stage 3: Feature Extraction")
    from extractor import extract_and_save_all
    extract_and_save_all(
        processed_dir=PROCESSED_DIR,
        features_dir=FEATURES_DIR,
        use_augmented=True,
        verbose=True,
    )


def stage_mrmr(K: int = 200):
    section(f"Stage 4: MRMR Selection (K={K})")
    from mrmr_selection import run_and_save
    run_and_save(K=K, features_dir=FEATURES_DIR, verbose=True)


def stage_knn():
    section("Stage 5: KNN")
    from collections import Counter

    train = np.load(os.path.join(FEATURES_DIR, 'train_mrmr.npz'))
    val   = np.load(os.path.join(FEATURES_DIR, 'val_mrmr.npz'))
    test  = np.load(os.path.join(FEATURES_DIR, 'test_mrmr.npz'))

    X_tr, y_tr = train['X'].astype(np.float32), train['y']
    X_val, y_val = val['X'].astype(np.float32), val['y']
    X_te, y_te = test['X'].astype(np.float32), test['y']

    mean = X_tr.mean(axis=0); std = X_tr.std(axis=0) + 1e-8
    X_tr  = (X_tr  - mean) / std
    X_val = (X_val - mean) / std
    X_te  = (X_te  - mean) / std

    best_k, best_acc = 1, 0
    for k in [1, 3, 5, 7, 9]:
        preds = []
        for x in X_val:
            dists = np.sqrt(((X_tr - x)**2).sum(axis=1))
            idx   = np.argsort(dists)[:k]
            preds.append(Counter(y_tr[idx]).most_common(1)[0][0])
        acc = float(np.mean(np.array(preds) == y_val))
        print(f"  k={k}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_k = acc, k

    # Test with best k
    preds = []
    for x in X_te:
        dists = np.sqrt(((X_tr - x)**2).sum(axis=1))
        idx   = np.argsort(dists)[:best_k]
        preds.append(Counter(y_tr[idx]).most_common(1)[0][0])
    test_acc = float(np.mean(np.array(preds) == y_te))
    print(f"\n  Best k={best_k}  Test Accuracy={test_acc:.4f}")

    from metrics import evaluate, classification_report
    results = evaluate(y_te, np.array(preds), class_names=CLASS_NAMES)
    print(classification_report(results))


def stage_softmax(optimizer='adam', epochs=100, batch_size=256,
                  lr=1e-3, l2=1e-4, patience=10):
    section(f"Stage 6: Softmax ({optimizer})")
    from softmax import SoftmaxClassifier

    train = np.load(os.path.join(FEATURES_DIR, 'train_mrmr.npz'))
    val   = np.load(os.path.join(FEATURES_DIR, 'val_mrmr.npz'))
    test  = np.load(os.path.join(FEATURES_DIR, 'test_mrmr.npz'))

    X_tr, y_tr = train['X'].astype(np.float32), train['y']
    X_val, y_val = val['X'].astype(np.float32),   val['y']
    X_te, y_te = test['X'].astype(np.float32),  test['y']

    mean = X_tr.mean(axis=0); std = X_tr.std(axis=0) + 1e-8
    X_tr  = (X_tr  - mean) / std
    X_val = (X_val - mean) / std
    X_te  = (X_te  - mean) / std

    model = SoftmaxClassifier(n_features=X_tr.shape[1], n_classes=N_CLASSES)
    model.fit(X_tr, y_tr, X_val, y_val,
              optimizer=optimizer, lr=lr, epochs=epochs,
              batch_size=batch_size, l2=l2, patience=patience,
              log_dir=LOGS_DIR)

    test_acc = model.accuracy(X_te, y_te)
    print(f"\n  Test Accuracy: {test_acc:.4f}")

    from metrics import evaluate, classification_report
    results = evaluate(y_te, model.predict(X_te), class_names=CLASS_NAMES)
    print(classification_report(results))


def stage_cnn(optimizer='adam', epochs=30, batch_size=64,
              lr=1e-3, l2=1e-4, patience=7):
    section(f"Stage 7: CNN from scratch ({optimizer})")
    from cnn_scratch import CNN

    train = np.load(os.path.join(PROCESSED_DIR, 'train_augmented.npz'), allow_pickle=True)
    val   = np.load(os.path.join(PROCESSED_DIR, 'val.npz'),  allow_pickle=True)
    test  = np.load(os.path.join(PROCESSED_DIR, 'test.npz'), allow_pickle=True)

    X_tr  = train['images'].transpose(0,3,1,2).astype(np.float32)
    y_tr  = train['labels'].astype(np.int32)
    X_val = val['images'].transpose(0,3,1,2).astype(np.float32)
    y_val = val['labels'].astype(np.int32)
    X_te  = test['images'].transpose(0,3,1,2).astype(np.float32)
    y_te  = test['labels'].astype(np.int32)

    H, W = X_tr.shape[2], X_tr.shape[3]
    model = CNN(input_shape=(3, H, W),
                conv_configs=[(16,3,1),(32,3,1)],
                pool_size=2, n_classes=N_CLASSES)

    model.fit(X_tr, y_tr, X_val, y_val,
              optimizer=optimizer, lr=lr, epochs=epochs,
              batch_size=batch_size, l2=l2, patience=patience,
              log_dir=LOGS_DIR)

    test_acc = model.accuracy(X_te, y_te)
    print(f"\n  Test Accuracy: {test_acc:.4f}")

    from metrics import evaluate, classification_report
    results = evaluate(y_te, model.predict(X_te), class_names=CLASS_NAMES)
    print(classification_report(results))


def stage_convmixer(epochs=50, batch_size=64, lr=1e-3, patience=10):
    section("Stage 8: ConvMixer")
    try:
        from convmixer_torch import train_convmixer
    except ImportError:
        from convmixer_keras import train_convmixer

    train_convmixer(
        processed_dir=PROCESSED_DIR,
        log_dir=LOGS_DIR,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_STAGES = ['preprocess', 'augment', 'features', 'mrmr',
              'knn', 'softmax', 'cnn', 'convmixer']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 2 end-to-end pipeline.")
    parser.add_argument('--stages', nargs='+', default=ALL_STAGES,
                        choices=ALL_STAGES,
                        help='Stages to run (default: all)')
    parser.add_argument('--size',       type=int,   default=150)
    parser.add_argument('--multiplier', type=int,   default=3)
    parser.add_argument('--K',          type=int,   default=200)
    parser.add_argument('--epochs-softmax', type=int, default=100)
    parser.add_argument('--epochs-cnn',     type=int, default=30)
    parser.add_argument('--epochs-convmixer', type=int, default=50)
    args = parser.parse_args()

    start = datetime.now()
    print(f"Pipeline start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stages: {args.stages}")

    stage_map = {
        'preprocess': lambda: stage_preprocess(args.size),
        'augment':    lambda: stage_augment(args.multiplier),
        'features':   stage_features,
        'mrmr':       lambda: stage_mrmr(args.K),
        'knn':        stage_knn,
        'softmax':    lambda: stage_softmax(epochs=args.epochs_softmax),
        'cnn':        lambda: stage_cnn(epochs=args.epochs_cnn),
        'convmixer':  lambda: stage_convmixer(epochs=args.epochs_convmixer),
    }

    for stage in args.stages:
        stage_map[stage]()

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\nPipeline complete in {elapsed:.1f} minutes.")