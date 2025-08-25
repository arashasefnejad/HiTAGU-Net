import os, csv, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='root/class_name/video_id/frames')
    ap.add_argument('--out', required=True, help='output CSV path')
    ap.add_argument('--label_map', required=False, help='optional text file: each line = class_name')
    args = ap.parse_args()

    label_map = {}
    if args.label_map and os.path.exists(args.label_map):
        with open(args.label_map, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                label_map[line.strip()] = i

    classes = sorted([d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))])
    rows = []
    for cls in classes:
        cpath = os.path.join(args.root, cls)
        label = label_map.get(cls, classes.index(cls))  # fallback to alphabetical index
        for vid in sorted(os.listdir(cpath)):
            vpath = os.path.join(cpath, vid)
            if not os.path.isdir(vpath):
                continue
            nframes = len([f for f in os.listdir(vpath) if f.lower().endswith('.jpg')])
            if nframes == 0:
                continue
            rel = os.path.relpath(vpath, args.root)
            rows.append([rel, label, nframes])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)

if __name__ == '__main__':
    main()
