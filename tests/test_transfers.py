"""Tests for src/scripts/transfers.py (WP14 download)."""

import transfers as tr

SENDER = "0x" + "11" * 20
RECIPIENT = "0x" + "22" * 20


def log(address, topics, data):
    return {"address": address, "topics": topics, "data": data, "blockNumber": "0x10", "logIndex": "0x2", "transactionHash": "0xab"}


def padded(address):
    return "0x" + "0" * 24 + address[2:]


def test_a_usdc_transfer_is_decoded_in_whole_tokens():
    row = tr.decode_transfer_log(log("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", [tr.TOPIC_TRANSFER, padded(SENDER), padded(RECIPIENT)], hex(2_500_000)))
    assert row["token"] == "USDC" and row["amount"] == 2.5
    assert row["sender"] == SENDER and row["recipient"] == RECIPIENT and row["block"] == 16


def test_other_contracts_and_non_standard_logs_are_skipped():
    assert tr.decode_transfer_log(log("0x" + "33" * 20, [tr.TOPIC_TRANSFER, padded(SENDER), padded(RECIPIENT)], "0x1")) is None
    assert tr.decode_transfer_log(log("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", [tr.TOPIC_TRANSFER, padded(SENDER)], "0x1")) is None
