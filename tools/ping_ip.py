import argparse
import platform
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Ping a range of IPs on a subnet")
    parser.add_argument("--subnet", "-S", required=True, help="First three octets, e.g. 192.168.15")
    parser.add_argument("--start", "-s", type=int, required=True, help="First host octet")
    parser.add_argument("--end", "-e", type=int, required=True, help="Last host octet")
    return parser.parse_args()


def ping(ip: str) -> bool:
    is_windows = platform.system().lower() == "windows"
    # Windows ping uses -n for packet count, Linux/macOS use -c
    count_flag = "-n" if is_windows else "-c"
    # Windows -w is a per-packet timeout in ms; Linux -w is a total deadline in seconds
    timeout = 1000 if is_windows else 5
    result = subprocess.run(
        ["ping", count_flag, "5", "-w", str(timeout), ip],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # ping exits 0 if at least one of the packets got a reply
    return result.returncode == 0


def main():
    args = parse_args()
    cnt = 0
    for i in range(args.start, args.end + 1):
        ip = f"{args.subnet}.{i}"
        if ping(ip):
            cnt += 1
        else:
            print(f"{ip} not connected")
    if cnt == args.end - args.start + 1:
        print('All connected')


if __name__ == "__main__":
    main()
