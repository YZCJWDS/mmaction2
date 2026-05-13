"""
检查 EGTEA Gaze+ 数据集完整性。
在云端解压数据后运行此脚本，确认视频文件与标注文件匹配。

Usage:
    python projects/egtea_gaze/tools/check_data.py \
        --data-root /root/data/egtea/videos \
        --ann-dir /root/data/egtea/annotations
"""
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Check EGTEA dataset integrity')
    parser.add_argument('--data-root', type=str,
                        default='/root/data/egtea/videos',
                        help='Root directory containing cropped_clips/')
    parser.add_argument('--ann-dir', type=str,
                        default='/root/data/egtea/action_annotation',
                        help='Directory containing train.txt, val.txt, test.txt')
    return parser.parse_args()


def check_split(ann_file, data_root, split_name):
    """Check one annotation split file."""
    if not os.path.exists(ann_file):
        print(f'[ERROR] {split_name}: annotation file not found: {ann_file}')
        return False

    with open(ann_file, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    missing = []
    labels = set()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        video_path = parts[0]
        label = int(parts[1])
        labels.add(label)

        full_path = os.path.join(data_root, video_path)
        if not os.path.exists(full_path):
            missing.append(video_path)

    found = total - len(missing)
    print(f'[{split_name}] Total: {total}, Found: {found}, '
          f'Missing: {len(missing)}, Labels: {len(labels)} '
          f'(range {min(labels)}-{max(labels)})')

    if missing:
        print(f'  First 5 missing files:')
        for m in missing[:5]:
            print(f'    {m}')

    return len(missing) == 0


def check_video_dir(data_root):
    """Check video directory structure."""
    clips_dir = os.path.join(data_root, 'cropped_clips')
    if not os.path.exists(clips_dir):
        print(f'[ERROR] cropped_clips directory not found: {clips_dir}')
        return False

    sessions = [d for d in os.listdir(clips_dir)
                if os.path.isdir(os.path.join(clips_dir, d))]
    total_videos = 0
    for session in sessions:
        session_dir = os.path.join(clips_dir, session)
        videos = [f for f in os.listdir(session_dir) if f.endswith('.mp4')]
        total_videos += len(videos)

    print(f'[Videos] Sessions: {len(sessions)}, Total clips: {total_videos}')
    return True


def main():
    args = parse_args()

    print('=' * 60)
    print('EGTEA Gaze+ Dataset Integrity Check')
    print('=' * 60)
    print(f'Data root: {args.data_root}')
    print(f'Ann dir:   {args.ann_dir}')
    print('-' * 60)

    # Check video directory
    check_video_dir(args.data_root)
    print('-' * 60)

    # Check each split
    all_ok = True
    for split in ['train', 'val', 'test']:
        ann_file = os.path.join(args.ann_dir, f'{split}.txt')
        ok = check_split(ann_file, args.data_root, split)
        if not ok:
            all_ok = False

    print('-' * 60)
    if all_ok:
        print('[PASS] All checks passed! Dataset is ready for training.')
    else:
        print('[FAIL] Some files are missing. Please check data extraction.')

    print('=' * 60)


if __name__ == '__main__':
    main()
