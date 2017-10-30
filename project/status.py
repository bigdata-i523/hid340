from bigchaindb_driver import BigchainDB
from bigchaindb_driver.crypto import generate_keypair

bdb_root_url = "https://test.ipdb.io"
tokens = {"app_id": "39c9142b", "app_key": "186598f642ed2b6fc4c649f44b1dbe07"}
bdb = BigchainDB("https://test.ipdb.io", headers=tokens)

txid = "7bfc3839eec44c9035b17e7d4ef7e9d959827761af7857620743998442ec68b6"
#status = bdb.transactions.status("7bfc3839eec44c9035b17e7d4ef7e9d959827761af7857620743998442ec68b6")

bdb.assets.get(search="bigchaindb")

