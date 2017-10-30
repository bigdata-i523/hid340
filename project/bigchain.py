# -*- coding: utf-8 -*-
"""
BigchainDB demo
"""

from bigchaindb_driver import BigchainDB
from bigchaindb_driver.crypto import generate_keypair

bdb_root_url = "https://test.ipdb.io"
tokens = {"app_id": "", "app_key": ""}
bdb = BigchainDB("https://test.ipdb.io", headers=tokens)

bicycle = {"data": {"bicycle": {"serial_number": "abcd1234", "maker": "Xyz"}}}
metadata = {"planet": "earth"}

alice, bob = generate_keypair(), generate_keypair()

prepared_tx = bdb.transactions.prepare(
    operation="CREATE",
    signers=alice.public_key,
    asset=bicycle,
    metadata=metadata        
)

fulfilled_tx = bdb.transactions.fulfill(
    prepared_tx, private_keys=alice.private_key
)

sent_tx = bdb.transactions.send(fulfilled_tx)
txid = fulfilled_tx["id"]
print(txid)

# 7bfc3839eec44c9035b17e7d4ef7e9d959827761af7857620743998442ec68b6
