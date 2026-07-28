#!/usr/bin/env python3
import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(
        description="Build PIM intra-chiplet delay profile from PopNet delayInfo file."
    )
    p.add_argument(
        "--delay",
        default="delayInfo_pim_intra.txt",
        help="Input PopNet delay info file.",
    )
    p.add_argument(
        "--out",
        default="reports/pim_intra_delay_profile.txt",
        help="Output delay profile file.",
    )
    p.add_argument(
        "--npu-count",
        type=int,
        default=16,
        help="Number of NPUs in PIM chiplet.",
    )
    return p.parse_args()


def to_int(token):
    # PopNet delay file should be integer. Keep float fallback for safety.
    try:
        return int(token)
    except Exception:
        return int(round(float(token)))


def load_delay_records(path):
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            try:
                cycle = to_int(parts[0])
                src = to_int(parts[1])
                dst = to_int(parts[2])
                desc = to_int(parts[3])
                delay_cnt = max(0, to_int(parts[4]))
            except Exception:
                continue
            if len(parts) < 5 + delay_cnt:
                continue
            delays = []
            ok = True
            for i in range(delay_cnt):
                try:
                    delays.append(to_int(parts[5 + i]))
                except Exception:
                    ok = False
                    break
            if not ok or not delays:
                continue
            records.append(
                {
                    "line": ln,
                    "cycle": cycle,
                    "src": src,
                    "dst": dst,
                    "desc": desc,
                    "delays": delays,
                }
            )
    return records


def infer_id_offset(records, npu_count):
    ids = set()
    for r in records:
        ids.add(int(r["src"]))
        ids.add(int(r["dst"]))
    if not ids:
        return 0
    min_id = min(ids)
    max_id = max(ids)
    if 0 <= min_id and max_id < npu_count:
        return 0
    # Typical merged case for PIM local nodes: [offset, offset+npu_count-1]
    offset = min_id
    if all(0 <= (x - offset) < npu_count for x in ids):
        return int(offset)
    return 0


def build_profile(records, npu_count):
    n = max(1, int(npu_count))
    id_offset = infer_id_offset(records, n)
    per_dst = [0 for _ in range(n)]
    per_src = [0 for _ in range(n)]
    packet_count = 0
    total_dst = 0
    total_src = 0

    for r in records:
        src = int(r["src"]) - id_offset
        dst = int(r["dst"]) - id_offset
        if src < 0 or src >= n or dst < 0 or dst >= n:
            continue
        delays = r["delays"]
        src_delay = int(delays[0]) if len(delays) >= 1 else 0
        dst_delay = int(delays[1]) if len(delays) >= 2 else src_delay
        per_src[src] += max(0, src_delay)
        per_dst[dst] += max(0, dst_delay)
        total_src += max(0, src_delay)
        total_dst += max(0, dst_delay)
        packet_count += 1

    return {
        "npu_count": n,
        "id_offset": int(id_offset),
        "packet_count": int(packet_count),
        "total_src_delay": int(total_src),
        "total_dst_delay": int(total_dst),
        "per_src_delay": per_src,
        "per_dst_delay": per_dst,
    }


def build_profile_with_local_controller(records, npu_count):
    n = max(1, int(npu_count))
    controller_local_id = n
    per_dst = [0 for _ in range(n)]
    per_src = [0 for _ in range(n)]
    packet_count = 0
    total_dst = 0
    total_src = 0

    for r in records:
        src = int(r["src"])
        dst = int(r["dst"])
        if src == controller_local_id or dst == controller_local_id:
            packet_count += 1
            delays = r["delays"]
            src_delay = int(delays[0]) if len(delays) >= 1 else 0
            dst_delay = int(delays[1]) if len(delays) >= 2 else src_delay
            total_src += max(0, src_delay)
            total_dst += max(0, dst_delay)
            continue
        if src < 0 or src >= n or dst < 0 or dst >= n:
            continue
        delays = r["delays"]
        src_delay = int(delays[0]) if len(delays) >= 1 else 0
        dst_delay = int(delays[1]) if len(delays) >= 2 else src_delay
        per_src[src] += max(0, src_delay)
        per_dst[dst] += max(0, dst_delay)
        total_src += max(0, src_delay)
        total_dst += max(0, dst_delay)
        packet_count += 1

    return {
        "npu_count": n,
        "id_offset": 0,
        "packet_count": int(packet_count),
        "total_src_delay": int(total_src),
        "total_dst_delay": int(total_dst),
        "per_src_delay": per_src,
        "per_dst_delay": per_dst,
        "controller_local_id": int(controller_local_id),
    }


def write_profile(path, profile):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# PIM intra delay profile generated from PopNet delayInfo\n")
        f.write("# format:\n")
        f.write("#   META <key> <value>\n")
        f.write("#   TOTAL <dst_delay_cycles>\n")
        f.write("#   PACKETS <count>\n")
        f.write("#   PE <id> <dst_delay_cycles> <src_delay_cycles>\n")
        f.write(f"META npu_count {profile['npu_count']}\n")
        f.write(f"META id_offset {profile['id_offset']}\n")
        f.write(f"TOTAL {profile['total_dst_delay']}\n")
        f.write(f"PACKETS {profile['packet_count']}\n")
        for pe in range(profile["npu_count"]):
            f.write(
                f"PE {pe} {int(profile['per_dst_delay'][pe])} {int(profile['per_src_delay'][pe])}\n"
            )


def main():
    args = parse_args()
    if not os.path.exists(args.delay):
        raise SystemExit(f"[pim-intra-delay] ERROR: delay file not found: {args.delay}")

    records = load_delay_records(args.delay)
    local_ids = set()
    for record in records:
        local_ids.add(int(record["src"]))
        local_ids.add(int(record["dst"]))
    if (args.npu_count in local_ids) and all(0 <= node_id <= args.npu_count for node_id in local_ids):
        profile = build_profile_with_local_controller(records, args.npu_count)
    else:
        profile = build_profile(records, args.npu_count)
    write_profile(args.out, profile)

    print(f"[pim-intra-delay] input: {args.delay}")
    print(f"[pim-intra-delay] packets: {profile['packet_count']}")
    print(
        f"[pim-intra-delay] total_dst_delay: {profile['total_dst_delay']}, "
        f"id_offset: {profile['id_offset']}"
    )
    print(f"[pim-intra-delay] output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
