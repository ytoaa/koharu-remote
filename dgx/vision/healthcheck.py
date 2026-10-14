import os
import httpx


def main():
    with httpx.Client(timeout=5, trust_env=False) as client:
        r = client.get(
            "http://127.0.0.1:8766/health",
            headers={"Authorization": "Bearer " + os.environ["VISION_API_TOKEN"]}
            if os.environ.get("VISION_API_TOKEN") else {},
        )
        r.raise_for_status()
        v = r.json()
        if not (
            type(v.get("protocol")) is int
            and v["protocol"] == 1
            and v.get("service") == "koharu-vision"
            and v.get("ready") is True
            and v.get("model_loaded") is True
        ):
            raise SystemExit("Vision not ready")


if __name__ == "__main__":
    main()
