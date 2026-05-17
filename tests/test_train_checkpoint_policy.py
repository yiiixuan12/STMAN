#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train


def main():
    calls = []

    def fake_save_checkpoint(path, model, optimizer, scheduler, epoch, monitor_value, config, state_override=None):
        calls.append(
            {
                "path": path,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "epoch": epoch,
                "monitor_value": monitor_value,
                "config": config,
                "state_override": state_override,
            }
        )

    original = train.save_checkpoint
    train.save_checkpoint = fake_save_checkpoint
    try:
        model = object()
        optimizer = object()
        scheduler = object()
        config = {"name": "stub"}

        train.save_named_checkpoint("last", "/tmp/last.pt", model, optimizer, scheduler, 7, 1.23, config)
        train.save_named_checkpoint("best", "/tmp/best.pt", model, optimizer, scheduler, 8, 0.99, config)
    finally:
        train.save_checkpoint = original

    assert len(calls) == 2, calls
    assert calls[0]["path"] == "/tmp/last.pt", calls[0]
    assert calls[0]["optimizer"] is optimizer, calls[0]
    assert calls[0]["scheduler"] is scheduler, calls[0]
    assert calls[1]["path"] == "/tmp/best.pt", calls[1]
    assert calls[1]["optimizer"] is None, calls[1]
    assert calls[1]["scheduler"] is None, calls[1]
    print("PASS test_train_checkpoint_policy")


if __name__ == "__main__":
    main()
