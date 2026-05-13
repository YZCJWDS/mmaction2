"""
将 EGTEA Gaze+ 原始标注转换为 MMAction2 VideoDataset 格式。

原始格式 (train_split1.txt):
    VideoName ActionID VerbID NounID
    例: P21-R04-ContinentalBreakfast-291175-294945-F006979-F007088 1 1 1

目标格式 (train.txt):
    cropped_clips/Session/VideoName.mp4 Label
    例: cropped_clips/P21-R04-ContinentalBreakfast/P21-R04-ContinentalBreakfast-291175-294945-F006979-F007088.mp4 0

注意: 原始 ActionID 从 1 开始，转换后 label 从 0 开始 (ActionID - 1)

Usage:
    python projects/egtea_gaze/tools/convert_annotations.py \
        --ann-dir /root/data/egtea/annotations \
        --split 1 \
        --video-dir /root/data/egtea/videos \
        --output-dir /root/data/egtea/annotations
"""
import argparse
import os
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert EGTEA annotations to MMAction2 format')
    parser.add_argument('--ann-dir', type=str,
                        default='/root/data/egtea/action_annotation',
                        help='Directory containing original split files')
    parser.add_argument('--split', type=int, default=1, choices=[1, 2, 3],
                        help='Which split to use (1, 2, or 3)')
    parser.add_argument('--video-dir', type=str,
                        default='/root/data/egtea/videos',
                        help='Video root directory (for validation)')
    parser.add_argument('--output-dir', type=str,
                        default='/root/data/egtea/action_annotation',
                        help='Output directory for converted files')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Ratio of training data to use as validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for val split')
    return parser.parse_args()


def extract_session(video_name):
    """
    从视频文件名中提取 session 名称。
    例: P21-R04-ContinentalBreakfast-291175-294945-F006979-F007088
    -> P21-R04-ContinentalBreakfast
    """
    # 匹配模式: OP/P + 数字 + -R + 数字 + -RecipeName
    match = re.match(r'((?:OP|P)\d+-R\d+-[A-Za-z]+)', video_name)
    if match:
        return match.group(1)
    return None


def convert_split_file(input_file, output_file, video_dir=None):
    """Convert one split file to MMAction2 format."""
    if not os.path.exists(input_file):
        print(f'  [SKIP] {input_file} not found')
        return []

    entries = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            video_name = parts[0]
            action_id = int(parts[1])
            # Convert to 0-indexed label
            label = action_id - 1

            session = extract_session(video_name)
            if session is None:
                print(f'  [WARN] Cannot parse session from: {video_name}')
                continue

            rel_path = f'cropped_clips/{session}/{video_name}.mp4'

            # Optionally validate file exists
            if video_dir:
                full_path = os.path.join(video_dir, rel_path)
                if not os.path.exists(full_path):
                    # Try without .mp4 extension in case name already has it
                    pass

            entries.append(f'{rel_path} {label}')

    with open(output_file, 'w') as f:
        f.write('\n'.join(entries) + '\n')

    print(f'  [OK] {output_file}: {len(entries)} entries')
    return entries


def main():
    args = parse_args()
    import random
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print(f'Converting EGTEA Gaze+ annotations (split {args.split})')
    print('=' * 60)

    # Convert train split
    train_input = os.path.join(args.ann_dir, f'train_split{args.split}.txt')
    train_output = os.path.join(args.output_dir, 'train.txt')
    val_output = os.path.join(args.output_dir, 'val.txt')

    print(f'\nProcessing training split: {train_input}')
    train_entries = convert_split_file(
        train_input,
        train_output,
        video_dir=args.video_dir)

    # Create validation split from training data
    if train_entries and args.val_ratio > 0:
        random.shuffle(train_entries)
        val_size = int(len(train_entries) * args.val_ratio)
        val_entries = train_entries[:val_size]
        train_entries_final = train_entries[val_size:]

        with open(train_output, 'w') as f:
            f.write('\n'.join(train_entries_final) + '\n')
        with open(val_output, 'w') as f:
            f.write('\n'.join(val_entries) + '\n')

        print(f'  Split into train ({len(train_entries_final)}) '
              f'and val ({len(val_entries)})')

    # Convert test split
    test_input = os.path.join(args.ann_dir, f'test_split{args.split}.txt')
    test_output = os.path.join(args.output_dir, 'test.txt')
    print(f'\nProcessing test split: {test_input}')
    convert_split_file(test_input, test_output, video_dir=args.video_dir)

    print('\n' + '=' * 60)
    print('Conversion complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
