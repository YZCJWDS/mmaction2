import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate MMAction2 VideoDataset annotations for EGTEA.')
    parser.add_argument(
        '--data-root',
        default='data/egtea',
        help='EGTEA root containing videos/ and annotations/.')
    parser.add_argument(
        '--split',
        default='1',
        choices=['1'],
        help='Official split id. Only split1 is used in the current stage.')
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio sampled from official train split.')
    parser.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='Random seed for reproducible val split generation.')
    return parser.parse_args()


def load_action_mapping(action_idx_path):
    action_id_to_name = {}
    with action_idx_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, action_id = line.rsplit(' ', 1)
            action_id_to_name[int(action_id)] = name
    return action_id_to_name


def build_video_index(video_root):
    stem_to_paths = defaultdict(list)
    for path in sorted(video_root.rglob('*.mp4')):
        rel_path = path.relative_to(video_root).as_posix()
        stem_to_paths[path.stem].append(rel_path)
    duplicates = {k: v for k, v in stem_to_paths.items() if len(v) > 1}
    return stem_to_paths, duplicates


def parse_split_file(split_path, split_name, stem_to_paths, action_id_to_name):
    samples = []
    bad_rows = []

    with split_path.open('r', encoding='utf-8') as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                bad_rows.append({
                    'split': split_name,
                    'line_no': line_no,
                    'reason': 'malformed_line',
                    'raw': raw_line.rstrip('\n')
                })
                continue

            clip_stem = parts[0]
            try:
                official_action_id = int(parts[1])
            except ValueError:
                bad_rows.append({
                    'split': split_name,
                    'line_no': line_no,
                    'reason': 'invalid_action_id',
                    'raw': raw_line.rstrip('\n')
                })
                continue

            matched_paths = stem_to_paths.get(clip_stem, [])
            if not matched_paths:
                bad_rows.append({
                    'split': split_name,
                    'line_no': line_no,
                    'reason': 'video_not_found',
                    'raw': raw_line.rstrip('\n')
                })
                continue
            if len(matched_paths) > 1:
                bad_rows.append({
                    'split': split_name,
                    'line_no': line_no,
                    'reason': 'duplicate_video_stem',
                    'raw': raw_line.rstrip('\n')
                })
                continue
            if official_action_id not in action_id_to_name:
                bad_rows.append({
                    'split': split_name,
                    'line_no': line_no,
                    'reason': 'missing_action_mapping',
                    'raw': raw_line.rstrip('\n')
                })
                continue

            samples.append({
                'clip_stem': clip_stem,
                'relative_path': matched_paths[0],
                'official_action_id': official_action_id,
                'action_name': action_id_to_name[official_action_id],
                'raw': raw_line.rstrip('\n')
            })
    return samples, bad_rows


def split_train_val(train_samples, val_ratio, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for sample in train_samples:
        grouped[sample['official_action_id']].append(sample)

    train_out = []
    val_out = []
    for action_id in sorted(grouped):
        items = grouped[action_id]
        rng.shuffle(items)
        if len(items) <= 1:
            val_count = 0
        else:
            val_count = round(len(items) * val_ratio)
            val_count = max(1, val_count)
            val_count = min(val_count, len(items) - 1)
        val_out.extend(items[:val_count])
        train_out.extend(items[val_count:])
    return train_out, val_out


def remap_labels(train_samples, val_samples, test_samples):
    used_action_ids = sorted({
        sample['official_action_id']
        for sample in train_samples + val_samples + test_samples
    })
    label_mapping = {
        official_action_id: new_label
        for new_label, official_action_id in enumerate(used_action_ids)
    }

    for sample in train_samples + val_samples + test_samples:
        sample['label'] = label_mapping[sample['official_action_id']]

    return label_mapping


def write_ann_file(path, samples):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for sample in samples:
            f.write(f"{sample['relative_path']} {sample['label']}\n")


def write_label_mapping(path, label_mapping, action_id_to_name):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        f.write('new_label\tofficial_action_id\taction_name\n')
        for official_action_id, new_label in sorted(
                label_mapping.items(), key=lambda x: x[1]):
            f.write(
                f'{new_label}\t{official_action_id}\t'
                f'{action_id_to_name[official_action_id]}\n')


def print_preview(title, samples, n=5):
    print(title)
    for sample in samples[:n]:
        print(f"  {sample['relative_path']} {sample['label']}")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    video_root = data_root / 'videos'
    ann_root = data_root / 'annotations'

    action_idx_path = ann_root / 'action_idx.txt'
    train_split_path = ann_root / f'train_split{args.split}.txt'
    test_split_path = ann_root / f'test_split{args.split}.txt'

    if not video_root.exists():
        raise FileNotFoundError(f'Video root not found: {video_root}')
    if not action_idx_path.exists():
        raise FileNotFoundError(f'Action mapping not found: {action_idx_path}')
    if not train_split_path.exists():
        raise FileNotFoundError(f'Train split not found: {train_split_path}')
    if not test_split_path.exists():
        raise FileNotFoundError(f'Test split not found: {test_split_path}')

    action_id_to_name = load_action_mapping(action_idx_path)
    stem_to_paths, duplicate_stems = build_video_index(video_root)

    train_samples_raw, train_bad = parse_split_file(
        train_split_path, 'train_split1', stem_to_paths, action_id_to_name)
    test_samples_raw, test_bad = parse_split_file(
        test_split_path, 'test_split1', stem_to_paths, action_id_to_name)

    train_samples, val_samples = split_train_val(
        train_samples_raw, args.val_ratio, args.seed)
    label_mapping = remap_labels(train_samples, val_samples, test_samples_raw)

    train_txt = ann_root / 'train.txt'
    val_txt = ann_root / 'val.txt'
    test_txt = ann_root / 'test.txt'
    mapping_txt = ann_root / 'label_mapping_generated.txt'

    write_ann_file(train_txt, train_samples)
    write_ann_file(val_txt, val_samples)
    write_ann_file(test_txt, test_samples_raw)
    write_label_mapping(mapping_txt, label_mapping, action_id_to_name)

    bad_rows = train_bad + test_bad

    print(f'video_root: {video_root}')
    print(f'total_mp4: {sum(len(v) for v in stem_to_paths.values())}')
    print(f'duplicate_stems_in_videos: {len(duplicate_stems)}')
    print(f'bad_rows: {len(bad_rows)}')
    if bad_rows:
        reason_counter = Counter(item['reason'] for item in bad_rows)
        print('bad_row_reasons:')
        for reason, count in sorted(reason_counter.items()):
            print(f'  {reason}: {count}')
        print('bad_row_examples:')
        for item in bad_rows[:10]:
            print(
                f"  [{item['split']}:{item['line_no']}] "
                f"{item['reason']} -> {item['raw']}")

    print(f'num_classes: {len(label_mapping)}')
    print(f'train_samples: {len(train_samples)}')
    print(f'val_samples: {len(val_samples)}')
    print(f'test_samples: {len(test_samples_raw)}')

    print_preview('train_preview:', train_samples)
    print_preview('val_preview:', val_samples)
    print_preview('test_preview:', test_samples_raw)


if __name__ == '__main__':
    main()
