"""Manage both Koharu services from the DGX checkout; vLLM stays external."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=("vision", "rorem", "qwen", "all"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.service == "all":
        if args.arguments not in [[command] for command in ("check", "up", "health", "logs", "stop")]:
            parser.error("all supports check, up, health, logs or stop")
        from manage_vision import check as vision_check, main as vision
        from manage_rorem import check as rorem_check, main as rorem
        from manage_qwen import check as qwen_check, main as qwen, active as qwen_active
        # The staged Qwen deployment replaces RORem; never start both inpainters.
        inpaint_check = qwen_check if qwen_active() else rorem_check
        inpaint = qwen if qwen_active() else rorem
        if args.arguments == ["up"]:
            vision_check()
            inpaint_check()
        vision(args.arguments)
        inpaint(args.arguments)
        return
    if args.service == "vision":
        from manage_vision import main as run
    elif args.service == "rorem":
        from manage_rorem import main as run
    else:
        from manage_qwen import main as run
    run(args.arguments)


if __name__ == "__main__":
    main()
